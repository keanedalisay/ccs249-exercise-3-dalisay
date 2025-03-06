import re

from collections import Counter

import wikipedia

from nltk import bigrams, trigrams
from nltk.tokenize import word_tokenize

class NGramModel:
  def __init__(self, text):
    self.text = text
    self.tokens = word_tokenize(text, language='english')
    self.bigram_counts = Counter(bigrams(self.tokens))
    self.trigram_counts = Counter(trigrams(self.tokens))
    self.unigram_counts = Counter(self.tokens)

  def __bigram_probabilities(self):
    bigram_probs = { bigram: round(count / self.unigram_counts[bigram[0]], 3) for bigram, count in self.bigram_counts.items() }
    return bigram_probs
  
  def __trigram_probabilities(self):
    trigram_probs = { trigram: round(count / self.bigram_counts[trigram[0], trigram[1]], 3) for trigram, count in self.trigram_counts.items() }
    return trigram_probs

  def generate_text(self, start_word, len = 10, n_gram = 'bigram'):
    current_word = start_word.lower()
    generated_text = [current_word]

    for _ in range(len):
      if (n_gram == 'bigram'):
        candidates = { k[1]: v for k, v in self.__bigram_probabilities().items() if k[0] == current_word }
      else:
        candidates = { k[2]: v for k, v in self.__trigram_probabilities().items() if k[0] == current_word or k[1] == current_word }

      next_word = max(candidates, key=candidates.get)
      generated_text.append(next_word)
      current_word = next_word
    
    return ' '.join(generated_text)
  
  def perplexity(self, gen_text, n_gram = 'bigram'):
    
    gen_tokens = word_tokenize(gen_text, language='english')
    if n_gram == 'bigram':
      all_bigram_probs = self.__bigram_probabilities()
      gen_token_probs = [all_bigram_probs.get((gen_tokens[i-1], gen_tokens[i])) for i in range(1, len(gen_tokens))]
    else:
      all_trigram_probs = self.__trigram_probabilities()
      gen_token_probs = [all_trigram_probs.get((gen_tokens[i-2], gen_tokens[i-1], gen_tokens[i])) for i in range(2, len(gen_tokens))]
    
    print(gen_token_probs)
    perplexity = gen_token_probs[0]
    for i in range(1, len(gen_token_probs)):
      if (gen_token_probs[i] is not None):
        perplexity *= gen_token_probs[i]

    return round(pow(perplexity, -1))


def main():
  page = wikipedia.page('attatchment theory')
  wiki_text = page.content[:8004].lower() # This is approximately 1159 words, including new lines.
  wiki_text = re.sub(r'[=]', ' ', wiki_text) # Remove headings

  ngram_model = NGramModel(wiki_text)

  # Use the below code to print the probabilities and counts of bigrams and trigrams
  # for trigram, prob in ngram_model.trigram_probabilities().items():
  #   print(f"P({trigram[2]} | {trigram[0]}, {trigram[1]}) = {prob: .4f}")
  #   print(f"C({trigram[0] + ', ' + trigram[1] + ', ' + trigram[2]}) = {ngram_model.trigram_counts[trigram]}")
  #   print(f"C({trigram[0] + ', ' + trigram[1]}) = {ngram_model.bigram_counts[trigram[0], trigram[1]]}\n")

  gen_text = ngram_model.generate_text('the', 10, 'trigram')
  print(gen_text)
  print(ngram_model.perplexity(gen_text, 'trigram'))


if __name__ == '__main__':
  main()