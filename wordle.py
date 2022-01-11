import numpy as np
from tqdm import tqdm

def scores(hyp, ref):
    hyp = np.array(list(hyp))
    ref = np.array(list(ref))
    matched = hyp == ref
    scores = np.zeros(len(hyp), dtype=np.int)
    scores[matched] = 2
    positions = np.arange(len(hyp))
    for letter, position in zip(hyp[~matched], positions[~matched]):
        for ref_letter, ref_position in zip(ref[~matched], positions[~matched]):
            if letter == ref_letter:
                scores[position] = 1
                matched[ref_position] = True
                break
    return scores
            
def get_filter(hyp, scores):
    def word_filter(word):
        for letter, score, position in zip(hyp, scores, np.arange(len(hyp))):
            if score == 0:
                if letter in word:
                    return False
            if score == 1:
                if letter not in word:
                    return False
                if word[position] == letter:
                    return False
            if score == 2:
                if word[position] != letter:
                    return False
        return True
    return word_filter

def asarray(wordlist):
    # convert list of words into array of integers representing characters
    return np.array(
        [[ord(x) for x in word] for word in wordlist]
    )

def score_to_int(score, axis=0):
    # bit shift and sum to map scores to single integers
    if type(score) == list:
        score = np.array(score)
    n = score.shape[axis]
    shifts = 2 * np.arange(n).reshape((1,) * axis + (-1,))
    return np.sum(score << shifts, axis=axis)

def int_to_score(x, depth=5):
    # bit shift and sum to map scores to single integers
    score = np.zeros(depth, dtype=np.int)
    for i in range(depth):
        score[i], x = x % 4, x >> 2
    return score

def npscores(guesses, solutions, convert_to_int=True):
    g = asarray(guesses)
    s = asarray(solutions)
    n = g.shape[1]
    g = g[:, np.newaxis, :]
    s = s[np.newaxis, :, :]
    correct = s == g
    scores = np.zeros(correct.shape, dtype=np.int)
    scores[correct] = 2 # correct letter and in the right spot
    matched = correct
    for i in np.arange(n):
        # check for correct letter in the wrong spot, but keep
        # track of what has already been matched
        other_positions = list(range(i)) + list(range(i+1, n))
        match = (g[:, :, i:i+1] == s[:, :, other_positions]) & \
                (~ matched[:, :, other_positions]) & \
                (scores[:, :, i:i+1] != 2)
        scores[:, :, i][match.any(axis=2)] = 1 
        matched[:, :, other_positions] |= match

    if convert_to_int:
        scores = score_to_int(scores, axis=2)
    return scores
                
def get_words():
    solutions = np.array([word.strip()
            for word in open('wordlist_solutions.txt').readlines()])
    guesses = np.array([word.strip()
            for word in open('wordlist_guesses.txt').readlines()])
    return solutions, guesses

def max_cluster(counts):
    return - max(counts)

def information(counts):
    p = counts / np.sum(counts)
    return - np.sum(p * np.log2(p))

def mean_cluster(counts):
    return - np.mean(counts)

def get_counts(score_matrix):
    return [np.unique(scores, return_counts=True)[1]
        for scores in score_matrix
    ]

class HumanPlayer:
    def __init__(self, solutions, guesses, metric=information):
        self.solutions = solutions

    def get_guess(self, clues):
        return input('guess: ')
        
class Player:
    def __init__(self, solutions, guesses, metric=information):
        self.solutions = solutions
        self.guesses = guesses
        self.metric = metric

        self.memo = {}
        
        self.solution_mapping = {word: i for i, word in enumerate(solutions)}
        self.guess_mapping = {word: i for i, word in enumerate(guesses)}
        self.score_matrix = npscores(guesses, solutions)

    def get_guess(self, clues):
        if tuple(clues) in self.memo:
            return self.memo[tuple(clues)]
        valid = np.full(shape=self.solutions.shape, fill_value=True)
        for guess, score in clues:
            i = self.guess_mapping[guess] 
            m = self.score_matrix[i, :] == score
            valid &= self.score_matrix[i, :] == score
        counts = get_counts(self.score_matrix[:, valid])
        values = np.array([self.metric(x) for x in counts])
        max_value = np.max(values)
        print('guess value: ', max_value)
        guesses = sorted(self.guesses[values == max_value], key=lambda x: x not
                in self.solutions[valid])
        self.memo[tuple(clues)] = guesses[0]
        return guesses[0]

class HumanScorer:

    def get_score(self, guess):
        score = input('score: ')
        score = score_to_int(list(int(x) for x in score))
        return score

class Scorer:
    def __init__(self, secret):
        self.secret = secret

    def get_score(self, guess):
        score = npscores([guess], [self.secret], convert_to_int=False).squeeze()
        print('score: ', score)
        return score_to_int(score)

def game(player, scorer):
    num_guesses = 0
    clues = []
    while num_guesses < 10:
        guess = player.get_guess(clues)
        num_guesses += 1
        print('guess: ', guess)
        score = scorer.get_score(guess)
        if score == 682:
            return num_guesses
        clues.append((guess, score))

if __name__ == '__main__':
    solutions, guesses = get_words()
    player = Player(solutions, guesses, metric=information)

    #game(player,HumanScorer())

    num_guesses = []
    for word in tqdm(solutions):
        scorer = Scorer(word)
        n = game(player, scorer)
        print(f'{word} took {n} guesses')
        num_guesses.append(n)
    max_guesses = np.max(num_guesses)
    n_worst_cases = sum(np.array(num_guesses) == max_guesses)
    mean_guesses = np.mean(num_guesses)
    
    print('player took {} guesses on average'.format(np.mean(num_guesses)))
    print('player took a maximum of {} guesses on {} words'.format(max_guesses, n_worst_cases))
    
