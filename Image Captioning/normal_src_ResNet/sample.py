import torch
import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image



images = ['./test_examples/me.jpg', './test_examples/boat.jpg', './test_examples/bus.jpg','./test_examples/child.jpg','./test_examples/dog.jpg','./test_examples/horse.jpg']
ENCODER_PATH = './flickr8k_models/encoder-10-558.ckpt'
DECODER_PATH = './flickr8k_models/decoder-10-558.ckpt'
VOCAB_PATH = './data/flickr8k_vocab.pkl'
IMG_SIZE = 224
EMBED_SIZE = 256
HIDDEN_SIZE = 384
NUM_LAYERS = 1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main():
  for image in images:
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(EMBED_SIZE).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    decoder.load_state_dict(torch.load(DECODER_PATH))

    # Prepare an image
    image = load_image(image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    #sampled_seqs = decoder.sample(feature)
    sampled_seqs = decoder.sample_beam_search(feature, vocab, device)
        
    for s in sampled_seqs:
        # Convert word_ids to words
        sampled_caption = []
        for word_id in s:
            word = vocab.idx2word[int(word_id)]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        # Print out the image and the generated caption
        print (sentence)
    
    #image = Image.open(args.image)
    #plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    main()

    