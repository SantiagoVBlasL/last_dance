#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convolutional_vae2.py (v2.0 con GroupNorm)

Implementación del VAE Convolucional.
Cambios:
- Reemplazado BatchNorm2d por GroupNorm para mayor estabilidad con
  diferentes tamaños de batch.
- Añadido el hiperparámetro 'num_groups' para controlar GroupNorm.
"""
from typing import Tuple, Union, List
import torch
import torch.nn as nn

__all__ = ["ConvolutionalVAE"]


class ConvolutionalVAE(nn.Module):
    """CNN‑based Variational Autoencoder con GroupNorm."""

    def __init__(
        self,
        input_channels: int = 6,
        latent_dim: int = 128,
        image_size: int = 131,
        final_activation: str = "tanh",
        intermediate_fc_dim_config: Union[int, str] = "0",
        dropout_rate: float = 0.2,
        use_layernorm_fc: bool = False,
        num_conv_layers_encoder: int = 4,
        decoder_type: str = "convtranspose",
        num_groups: int = 8,  # ⬅️ NUEVO: Hiperparámetro para GroupNorm
    ) -> None:
        super().__init__()

        # Validaciones y asignaciones de parámetros
        if num_conv_layers_encoder not in {3, 4}:
            raise ValueError("num_conv_layers_encoder must be 3 or 4.")
        if decoder_type not in {"upsample_conv", "convtranspose"}:
            raise ValueError("decoder_type must be 'upsample_conv' or 'convtranspose'.")

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.final_activation_name = final_activation
        self.dropout_rate = dropout_rate
        self.use_layernorm_fc = use_layernorm_fc
        self.num_conv_layers_encoder = num_conv_layers_encoder
        self.decoder_type = decoder_type
        self.num_groups = num_groups  # ⬅️ NUEVO

        # ------------------------------
        # Encoder (conv → optional FC)
        # ------------------------------
        encoder_layers: List[nn.Module] = []
        curr_ch = input_channels
        base_conv_ch = [max(16, input_channels * 2), max(32, input_channels * 4),
                        max(64, input_channels * 8), max(128, input_channels * 16)]
        conv_ch_enc = [min(c, 256) for c in base_conv_ch][: num_conv_layers_encoder]
        kernels = [7, 5, 5, 3][: num_conv_layers_encoder]
        paddings = [1, 1, 1, 1][: num_conv_layers_encoder]
        strides = [2, 2, 2, 2][: num_conv_layers_encoder]

        spatial_dims = [image_size]
        dim = image_size
        for k, p, s, ch_out in zip(kernels, paddings, strides, conv_ch_enc):
            encoder_layers += [
                nn.Conv2d(curr_ch, ch_out, kernel_size=k, stride=s, padding=p),
                nn.GELU(),
                # --- CAMBIO PRINCIPAL ---
                nn.GroupNorm(self.num_groups, ch_out), # ⬅️ CAMBIADO de BatchNorm2d
                nn.Dropout2d(p=dropout_rate),
            ]
            curr_ch = ch_out
            dim = ((dim + 2 * p - k) // s) + 1
            spatial_dims.append(dim)
        self.encoder_conv = nn.Sequential(*encoder_layers)

        self.final_conv_ch = curr_ch
        self.final_spatial_dim = dim
        flat_size = curr_ch * dim * dim

        # Capas FC del codificador (sin cambios, LayerNorm es mejor aquí)
        self.intermediate_fc_dim = self._resolve_intermediate_fc(intermediate_fc_dim_config, flat_size)
        if self.intermediate_fc_dim:
            fc_layers = [nn.Linear(flat_size, self.intermediate_fc_dim)]
            if use_layernorm_fc:
                fc_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            fc_layers += [nn.GELU(), nn.BatchNorm1d(self.intermediate_fc_dim), nn.Dropout(p=dropout_rate)]
            self.encoder_fc_intermediate = nn.Sequential(*fc_layers)
            mu_logvar_in = self.intermediate_fc_dim
        else:
            self.encoder_fc_intermediate = nn.Identity()
            mu_logvar_in = flat_size

        self.fc_mu = nn.Linear(mu_logvar_in, latent_dim)
        self.fc_logvar = nn.Linear(mu_logvar_in, latent_dim)

        # ------------------------------
        # Decoder (latent → conv‑transpose)
        # ------------------------------
        if self.intermediate_fc_dim:
            dec_fc_layers = [nn.Linear(latent_dim, self.intermediate_fc_dim)]
            if use_layernorm_fc:
                dec_fc_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            dec_fc_layers += [nn.GELU(), nn.BatchNorm1d(self.intermediate_fc_dim), nn.Dropout(p=dropout_rate)]
            self.decoder_fc_intermediate = nn.Sequential(*dec_fc_layers)
            dec_fc_out = self.intermediate_fc_dim
        else:
            self.decoder_fc_intermediate = nn.Identity()
            dec_fc_out = latent_dim

        self.decoder_fc_to_conv = nn.Linear(dec_fc_out, flat_size)

        decoder_layers: List[nn.Module] = []
        if decoder_type == "convtranspose":
            curr_ch_dec = self.final_conv_ch
            target_conv_t_channels = conv_ch_enc[-2 :: -1] + [input_channels]
            decoder_kernels = kernels[::-1]
            decoder_paddings = paddings[::-1]
            decoder_strides = strides[::-1]

            output_paddings: List[int] = []
            tmp_dim = self.final_spatial_dim
            for i in range(num_conv_layers_encoder):
                k, s, p = decoder_kernels[i], decoder_strides[i], decoder_paddings[i]
                target_dim = spatial_dims[num_conv_layers_encoder - 1 - i]
                op = target_dim - ((tmp_dim - 1) * s - 2 * p + k)
                op = max(0, min(s - 1, op))
                output_paddings.append(op)
                tmp_dim = (tmp_dim - 1) * s - 2 * p + k + op

            for i, ch_out in enumerate(target_conv_t_channels):
                decoder_layers += [
                    nn.ConvTranspose2d(
                        curr_ch_dec,
                        ch_out,
                        kernel_size=decoder_kernels[i],
                        stride=decoder_strides[i],
                        padding=decoder_paddings[i],
                        output_padding=output_paddings[i],
                    ),
                    nn.GELU() if i < len(target_conv_t_channels) - 1 else nn.Identity(),
                ]
                if i < len(target_conv_t_channels) - 1:
                    # --- CAMBIO PRINCIPAL ---
                    decoder_layers += [
                        nn.GroupNorm(self.num_groups, ch_out), # ⬅️ CAMBIADO de BatchNorm2d
                        nn.Dropout2d(p=dropout_rate)
                    ]
                curr_ch_dec = ch_out
        else:
            raise NotImplementedError("'upsample_conv' decoder not implemented yet.")

        if final_activation == "sigmoid":
            decoder_layers.append(nn.Sigmoid())
        elif final_activation == "tanh":
            decoder_layers.append(nn.Tanh())

        self.decoder_conv = nn.Sequential(*decoder_layers)

    # Resto de la clase (forward, encode, decode, etc.) no necesita cambios
    def _resolve_intermediate_fc(self, cfg: Union[int, str], flat_size: int) -> int:
        if cfg == "0" or cfg == 0: return 0
        if isinstance(cfg, str):
            cfg = cfg.lower()
            if cfg == "half": return flat_size // 2
            if cfg == "quarter": return flat_size // 4
            try: return int(cfg)
            except ValueError: return 0
        return int(cfg)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = self.encoder_fc_intermediate(h)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc_intermediate(z)
        h = self.decoder_fc_to_conv(h)
        h = h.view(h.size(0), self.final_conv_ch, self.final_spatial_dim, self.final_spatial_dim)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        if recon_x.shape != x.shape:
            recon_x = nn.functional.interpolate(recon_x, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        return recon_x, mu, logvar, z
