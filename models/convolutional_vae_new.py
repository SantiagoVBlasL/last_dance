import torch
import torch.nn as nn
from typing import List, Tuple, Union

class ConvolutionalVAE(nn.Module):
    """
    CNN-based Variational Autoencoder with
    • GroupNorm + LeakyReLU en vez de BatchNorm + ReLU
    • Skip-connections tipo U-Net (concatenación) entre encoder y decoder
    """

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
        norm_groups: int = 8,              # <- NEW: groupnorm groups
        use_skip: bool = True,             # <- NEW: enable / disable skips
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _gn(num_ch: int) -> nn.Module:          # GroupNorm fábrica
            g = min(norm_groups, num_ch) if num_ch >= norm_groups else 1
            return nn.GroupNorm(g, num_ch)

        act = lambda: nn.LeakyReLU(0.1, inplace=True)

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        curr_ch = input_channels
        base = [max(16, 2*input_channels),
                max(32, 4*input_channels),
                max(64, 8*input_channels),
                max(128,16*input_channels)]
        conv_ch_enc = [min(c, 256) for c in base][:num_conv_layers_encoder]
        kernels, pads, strides = [7,5,5,3][:num_conv_layers_encoder], [1]*4, [2]*4

        self.encoder_blocks: nn.ModuleList = nn.ModuleList()
        dim = image_size
        for k,p,s,ch_out in zip(kernels, pads, strides, conv_ch_enc):
            block = nn.Sequential(
                nn.Conv2d(curr_ch, ch_out, kernel_size=k, stride=s, padding=p),
                act(),
                _gn(ch_out),
                nn.Dropout2d(dropout_rate),
            )
            self.encoder_blocks.append(block)
            curr_ch = ch_out
            dim = ((dim + 2*p - k)//s) + 1

        self.final_conv_ch, self.final_spatial_dim = curr_ch, dim
        flat = curr_ch * dim * dim

        # --------- FC compresor opcional ----------
        self.inter_fc_dim = self._resolve_fc(intermediate_fc_dim_config, flat)
        if self.inter_fc_dim:
            fc_enc = [nn.Linear(flat, self.inter_fc_dim)]
            if use_layernorm_fc:
                fc_enc.append(nn.LayerNorm(self.inter_fc_dim))
            fc_enc += [act(), nn.BatchNorm1d(self.inter_fc_dim),
                       nn.Dropout(dropout_rate)]
            self.encoder_fc = nn.Sequential(*fc_enc)
            mu_input = self.inter_fc_dim
        else:
            self.encoder_fc = nn.Identity()
            mu_input = flat

        self.fc_mu     = nn.Linear(mu_input, latent_dim)
        self.fc_logvar = nn.Linear(mu_input, latent_dim)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        if self.inter_fc_dim:
            fc_dec = [nn.Linear(latent_dim, self.inter_fc_dim)]
            if use_layernorm_fc:
                fc_dec.append(nn.LayerNorm(self.inter_fc_dim))
            fc_dec += [act(), nn.BatchNorm1d(self.inter_fc_dim),
                       nn.Dropout(dropout_rate)]
            self.decoder_fc = nn.Sequential(*fc_dec)
            dec_in = self.inter_fc_dim
        else:
            self.decoder_fc = nn.Identity()
            dec_in = latent_dim

        self.fc_to_conv = nn.Linear(dec_in, flat)

        # ConvTranspose decoder with dynamic skip concatenation
        if decoder_type != "convtranspose":
            raise NotImplementedError

        self.decoder_blocks: nn.ModuleList = nn.ModuleList()
        rev_ch = conv_ch_enc[::-1]        # e.g. [128,64,32,16] → depende del nº capas
        target_chs = rev_ch[1:] + [input_channels]

        dec_k = kernels[::-1]
        dec_s = strides[::-1]
        dec_p = pads[::-1]

        tmp_dim = self.final_spatial_dim
        out_paddings: List[int] = []
        for k,s,p in zip(dec_k, dec_s, dec_p):
            tgt = (tmp_dim-1)*s - 2*p + k
            op = int((tmp_dim*s - tmp_dim + k - 2*p) % s)  # matemático
            out_paddings.append(op)
            tmp_dim = tgt + op

        curr_dec_ch = self.final_conv_ch
        for i, ch_out in enumerate(target_chs):
            layers = [
                nn.ConvTranspose2d(curr_dec_ch, ch_out,
                                   kernel_size=dec_k[i], stride=dec_s[i],
                                   padding=dec_p[i], output_padding=out_paddings[i]),
            ]
            if i < len(target_chs)-1:
                layers += [act(), _gn(ch_out), nn.Dropout2d(dropout_rate)]
            self.decoder_blocks.append(nn.Sequential(*layers))
            curr_dec_ch = ch_out

        self.use_skip = use_skip

        # output activation
        if final_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif final_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

    # ------------------------------------------------------------------
    # util
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_fc(cfg: Union[int,str], flat: int) -> int:
        if str(cfg) in {"0","none"}: return 0
        if isinstance(cfg,str):
            cfg = cfg.lower()
            if cfg=="half": return flat//2
            if cfg=="quarter": return flat//4
            return int(cfg)
        return int(cfg)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor, list]:
        skips = []
        h = x
        for blk in self.encoder_blocks:
            h = blk(h)
            if self.use_skip:
                skips.append(h)
        h_flat = h.view(h.size(0), -1)
        h_flat = self.encoder_fc(h_flat)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat), skips[::-1]  # reverse for decoder

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5*logvar)

    def decode(self, z: torch.Tensor, skips: list) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = self.fc_to_conv(h)
        h = h.view(h.size(0), self.final_conv_ch,
                   self.final_spatial_dim, self.final_spatial_dim)

        for i, blk in enumerate(self.decoder_blocks):
            h = blk(h)
            if self.use_skip and i < len(skips):
                # concat along channel dim
                h = torch.cat([h, skips[i]], dim=1)

        return self.out_act(h)

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        if recon.shape != x.shape:  # safeguard
            recon = nn.functional.interpolate(recon,
                                              size=(x.shape[2], x.shape[3]),
                                              mode="bilinear",
                                              align_corners=False)
        return recon, mu, logvar, z
