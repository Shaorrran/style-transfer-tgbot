import bot

# These imports are required for pickle (used in torch.load) to find all the classes needed
from architectures.model import StyleTransferCNN
from architectures.normalized_vgg import NormalizedVGG
from architectures.decoder import Decoder
from architectures.kmeans import KMeans

if __name__ == "__main__":
    bot.internals.start_bot()