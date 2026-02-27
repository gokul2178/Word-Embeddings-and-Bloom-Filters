from tqdm import tqdm
import sys
from nltk.corpus import stopwords, wordnet
import contextlib
import numpy as np
import scipy
import contextlib
import string
import nltk
import json
import copy
import spacy
import lemminflect
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print(os.getcwd())

# Try to import CuPy for GPU acceleration, fallback to NumPy if unavailable
try:
    import cupy as cp
    USE_GPU = True
    print("✓ GPU detected and initialized\n")
except Exception as e:
    print(f"⚠ GPU unavailable ({str(e)}), using CPU instead\n")
    cp = np
    USE_GPU = False

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
POS = ("CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", 
       "NNP", "NNPS", "NNS", "PDT", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "VB", 
       "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB")

def lemmatize(word): # Takes a word and uses the spacy lemmatizer to return the lemmatized form
    token = nlp(str(word))[0]
    lemma = token.lemma_
    inflections = {token._.inflect(pos) for pos in POS} # returns the inflection of the lemmatized token. (ex: run -> {'ran', 'run', 'runner', 'runnest', 'running', 'runs'} )
    return lemma, inflections

def tokenize(sentence): # Tokenizes a sentence and lemmatizes the words within
    tokenized = nlp(sentence.translate(str.maketrans('', '', string.punctuation)))
    return [token.lemma_ for token in tokenized 
            if token.lemma_.lower() not in en_stopwords 
            and wordnet.synsets(token.lemma_)] # disregards lemmatized token if it's in list of stopwords or not in english dictionary (wordnet)

nltk.download('stopwords')
nltk.download('wordnet')
en_stopwords = set(stopwords.words('english'))

with open('data/fairytales_word_tf-idfs.json', 'r') as f:
    tf_idfs = json.load(f)
with open('data/fairytales_word_bloom-filters.json', 'r') as f:
    bloom_filters = json.load(f)
with open('data/fairytales_tokenized.json', 'r') as f:
    tokenized_corpus = json.load(f)

iterative_vectors = {}

def rescale_bloom_filter(): # Rescales bloom filters to be in range [-1, 1] instead of [0, 1]
    for word in bloom_filters.keys():
        bloom_filters[word] = cp.array(bloom_filters[word], dtype=int) * 2 - 1

def generate_vector(word, tokenized_sentence, bits, deltas, iteration):
    """ 
    Generates vector representation for word when given a sentence.
    Uses GPU (CuPy) for accelerated computation.
    
    Args:
        word (str): The word to sum the neighbors of. Note that multiple instances of the word could occur.
        tokenized_sentence (list): The sentence the word instance(s) is/are contained in, as a list of tokens.
        bits (int): The number of bits the representation should be.
        deltas (int): The index of the neighbors, relative to the position of the target word(s), to sum. (e.g. [-4, -3, -2, -1, 1, 2, 3, 4] means the 4 words before and after the word(s)).
        iteration (int): The iteration. Although an integer input, indicates whether a previous iteration has occurred. If not, we will use the bloom filters of each word as their representations. Otherwise, we will use the representations from the previous iteration.
        
    Returns:
        instance_representation (cp.ndarray): The vector representation (on GPU) prior to averaging by the number of neighbors.
        adjacent_words (int): The number of adjacent words that were found and used to construct the representation.
    """
    indices = [i for i, x in enumerate(tokenized_sentence) if x == word] # Gets the indices of all occurrences of the word in this sentence.
    instance_representation = cp.zeros(bits) 
    adjacent_words = 0

    for index in indices: # for each occurrence of the word in the sentence
        for delta in deltas: # for each neighbor
            if index + delta < 0: # if the neighbor index is negative, skip.
                continue
            with contextlib.suppress(IndexError): # suppress IndexError if the neighbor is out of bounds
                adjacent_word = tokenized_sentence[index + delta]
                try: # if the neighbor does not have a tf-idf for this word on file, it is too infrequent to be relevant, so we skip and default to 0.
                    tf_idf = tf_idfs[word][adjacent_word]
                except KeyError:
                    tf_idf = 0
                try: # if the neighbor word doesn't have a representation on file, skip it
                    if iteration: # if this is not the first iteration, we use the preassigned iterative vectors for the adjacent word.
                        instance_representation += cp.array(preassign_iterative_vectors[adjacent_word]) * tf_idf
                    else:
                        instance_representation += bloom_filters[adjacent_word] * tf_idf
                    adjacent_words += 1
                except KeyError:
                    continue
    return instance_representation, adjacent_words

def extract_vectors(word, iteration, deltas=None, bits=32):
    """ 
        Extracts the vector representation of a word using GPU acceleration.
        
        Args:
            word (str): The word we are finding the representation of.
            iteration (int): The index of the current iteration.
            deltas (int): The index of the neighbors, relative to the position of the target word(s), to sum. (e.g. [-4, -3, -2, -1, 1, 2, 3, 4] means the 4 words before and after the word(s)).
            bits (int): The number of bits the representation should be.
    """
    if deltas is None: # default delta values, necessary to specify here because a list cannot be used as a default argument.
        deltas = [-4, -3, -2, -1, 1, 2, 3, 4]

    total_adjacent_words = 0
    representations = cp.zeros(bits)

    for sentence in tokenized_corpus:
        if word in sentence: # if the word is in the sentence, we pass it to the generate_vector function.
            representation, adjacent_words = generate_vector(word, sentence, bits, deltas, iteration)
            representations += representation # the representation accumulates to be the sum of all the neighbor representations.
            total_adjacent_words += adjacent_words # the count of neighbors accumulates
    if total_adjacent_words == 0:
        return representations  # return zeros if no neighbors found
    return representations / total_adjacent_words # we take the average of all neighbors by dividing the sum of their represntations by the count of neighbors.


def update_encoding(word, iteration, args):
    """Replaces the previous vector representation of word in iterative_vectors with the new one.
    Stores as GPU array for faster subsequent operations.
    """
    vector = extract_vectors(word, iteration, **args)
    iterative_vectors[word] = vector

def normalize_vector():
    """This is not used in the current implementation, but it normalizes the vectors to unit length using GPU acceleration.
    """
    for word in iterative_vectors.keys():
        iterative_vectors[word] = iterative_vectors[word] / cp.linalg.norm(iterative_vectors[word]) # normalized on GPU

def normalize_vector_dimensions(iterative_vectors):
    """Normalizes vector dimensions by (1) normalizing the length of each vector to 1 and (2) normalizing vectors along the dimensions (columns) using Robust Scaling to ignore outliers while simultaneously adjusting the scale of each dimension.
    Uses GPU acceleration with CuPy.
    """
    # Convert to GPU arrays
    vectors = cp.array([cp.asnumpy(iterative_vectors[word]) if isinstance(iterative_vectors[word], cp.ndarray) 
                        else iterative_vectors[word] for word in iterative_vectors.keys()])

    # Row normalization (GPU)
    norms = cp.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    # Column normalization (robust scaling on GPU)
    med = cp.median(vectors, axis=0)
    # CuPy equivalent of scipy.stats.iqr
    q75 = cp.percentile(vectors, 75, axis=0)
    q25 = cp.percentile(vectors, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1
    vectors = (vectors - med) / iqr

    return {
        word: vectors[i] for i, word in enumerate(iterative_vectors.keys())
    }


def sigmoid_normalize_vectors():
    """Not used in current implementation. GPU-accelerated sigmoid normalization.
    """
    for word in iterative_vectors.keys():
        iterative_vectors[word] = 2 / (1 + cp.exp(-iterative_vectors[word])) - 1  # sigmoid function + scale to pos/neg on GPU

def convert_to_cpu(iterative_vectors):
    """Convert GPU arrays back to CPU for JSON serialization.
    """
    cpu_vectors = {}
    for word, vector in iterative_vectors.items():
        if isinstance(vector, cp.ndarray):
            cpu_vectors[word] = cp.asnumpy(vector).tolist()
        else:
            cpu_vectors[word] = vector
    return cpu_vectors

if __name__ == '__main__':
    import os
    ITERATIONS = 400 # some amount of iterations, around 200 should be sufficient currently to observe the periodicity.
    NEIGHBORHOOD_SIZE = 4 # number of words to the left and right to consider as neighbors
    # Create deltas from -x to x excluding 0
    x = NEIGHBORHOOD_SIZE
    deltas = []
    deltas = [i for i in range(-NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE + 1) if i != 0]

    print(f"Using {'GPU (CuPy)' if USE_GPU else 'CPU (NumPy)'} for acceleration")
    print(f"Deltas: {deltas}")

    # Create output directory if it doesn't exist
    os.makedirs('data/iterative_vectors', exist_ok=True)

    rescale_bloom_filter()
    for i in range(ITERATIONS):
        preassign_iterative_vectors = copy.deepcopy(iterative_vectors)
        for word in tqdm(list(tf_idfs.keys()), desc=f"Iteration {i}/{ITERATIONS}", dynamic_ncols=True, leave=True, file=sys.stdout, ascii=True): # tqdm just gives fancy progress bar
            update_encoding(word, i, {'deltas': deltas, 'bits':32})
        iterative_vectors = normalize_vector_dimensions(iterative_vectors)
        
        # Convert GPU arrays to CPU for JSON serialization
        cpu_vectors = convert_to_cpu(iterative_vectors)
        with open(f'data/iterative_vectors/window_{NEIGHBORHOOD_SIZE}_iter_{i}.json', 'w+') as f:
            json.dump(cpu_vectors, f, indent=4) # saves file for each iteration for future reference