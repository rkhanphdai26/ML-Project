import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class AMFIMultimodalModel(nn.Module):
    def __init__(self, sar_channels=2, sensor_features=4, hidden_size=64):
        super(AMFIMultimodalModel, self).__init__()

        # 1. SPATIAL BRANCH (The "Eye")
        self.sar_branch = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=sar_channels,
            classes=hidden_size,
            decoder_attention_type="scse"
        )

        # 2. TEMPORAL BRANCH (The "Ear")
        self.sensor_branch = nn.GRU(
            input_size=sensor_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.temp_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True
        )

        # 3. MULTIMODAL FUSION
        # Batch Norm added here to stabilize the 192 incoming channels (64 + 128)
        self.fusion_bn = nn.BatchNorm2d(hidden_size + (hidden_size * 2))

        self.fusion_conv = nn.Conv2d(
            in_channels=hidden_size + (hidden_size * 2),
            out_channels=64,
            kernel_size=1
        )

        # Final prediction layer (Logits output)
        self.final_mask = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, sar_img, sensor_seq):
        # Step A: SAR Features
        sar_feats = self.sar_branch(sar_img)

        # Step B: Sensor Features
        gru_out, _ = self.sensor_branch(sensor_seq)
        attn_out, _ = self.temp_attn(gru_out, gru_out, gru_out)
        sensor_vector = attn_out[:, -1, :].unsqueeze(-1).unsqueeze(-1)

        # Step C: Cross-Modal Fusion
        sensor_context = sensor_vector.expand(-1, -1, sar_feats.size(2), sar_feats.size(3))
        combined = torch.cat([sar_feats, sensor_context], dim=1)

        # --- Stability Shield ---
        combined = self.fusion_bn(combined)

        out = torch.relu(self.fusion_conv(combined))

        # Step D: Final Output (Raw Logits for BCEWithLogitsLoss)
        return self.final_mask(out)