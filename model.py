
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(  # Extrae características mediante convoluciones.
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels), # Normaliza las activaciones de cada canal durante el entrenamiento.
            nn.ReLU(inplace=True) # Introduce una no linealidad y convierte los valores negativos en cero (funcion de activacion ReLU)
        )

    def forward(self, x):
        return self.block(x)



class CNNBlocks(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            CNNBlock(in_channels, out_channels),
            CNNBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = CNNBlocks(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        down = self.pool(features)

        return features, down


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()

        self.enc1 = EncoderBlock( 
            in_channels, base_channels #16
        )

        self.enc2 = EncoderBlock(
            base_channels, base_channels * 2 #32
        )

        self.enc3 = EncoderBlock(
            base_channels * 2, base_channels * 4 #64
        )

        self.enc4 = EncoderBlock(
            base_channels * 4, base_channels * 8 #128 
        )

    def forward(self, x):  #Caracteristicas que se gurdan para skip connections
        features1, x = self.enc1(x)
        features2, x = self.enc2(x)
        features3, x = self.enc3(x) 
        features4, x = self.enc4(x)

        return features1, features2, features3, features4, x  #X para salida a bottleneck


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = CNNBlocks( #reutiliza la clase CNNBlocks para aplicar dos bloques convolucionales consecutivos.
            in_channels,
            out_channels
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #Aumentar la resolución y reducir canales
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels, #tensor de entrada con 256 canales 
            out_channels=out_channels, #tensor de salida con 128 canales 
            kernel_size=2,
            stride=2
        )

        #Combinar decoder con características del encoder
        self.conv = CNNBlocks(
            in_channels=out_channels * 2,
            out_channels=out_channels
        )

    def forward(self, x, skip):
        # Aumentamos la resolución
        x = self.upsample(x)

        # Concatena por dimensión de canales 128 + 128 = 256
        x = torch.cat((x, skip), dim=1)

        # Procesa características combinadas
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()

        # 1. Encoder: extrae y guarda características
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels
        )

        # 2. Bottleneck: procesa las características más profundas
        self.bottleneck = Bottleneck(
            in_channels=base_channels * 8,
            out_channels=base_channels * 16
        )

        # 3. Decoder: recupera resolución usando skip connections
        self.dec4 = DecoderBlock(
            in_channels=base_channels * 16,
            out_channels=base_channels * 8
        )

        self.dec3 = DecoderBlock(
            in_channels=base_channels * 8,
            out_channels=base_channels * 4
        )

        self.dec2 = DecoderBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2
        )

        self.dec1 = DecoderBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels
        )

        # 4. Convierte los canales finales en un canal de predicción
        self.final_conv = nn.Conv2d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        # Encoder: guardamos las características para las skip connections
        features1, features2, features3, features4, x = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: de menor a mayor resolución
        x = self.dec4(x, features4)
        x = self.dec3(x, features3)
        x = self.dec2(x, features2)
        x = self.dec1(x, features1)

        # Predicción final por píxel
        x = self.final_conv(x)

        return x





if __name__ == "__main__":
    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=16
    )

    # Simulamos un batch de 4 imágenes RGB de 256 x 256
    x = torch.randn(4, 3, 256, 256)

    salida = model(x)

    print("Entrada:", x.shape)
    print("Salida de UNet:", salida.shape)
    print("Parámetros entrenables:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))



