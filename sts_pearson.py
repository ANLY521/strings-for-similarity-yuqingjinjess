from scipy.stats import pearsonr
import argparse
from util import parse_sts
from sts_nist import symmetrical_nist
from nltk import word_tokenize
from nltk import edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import pearsonr
from difflib import SequenceMatcher
from Levenshtein import distance

def symmetrical_bleu(text_pair):

    t1, t2 = text_pair
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())

    try:
        bleu_1 = sentence_bleu([t1_toks, ], t2_toks, smoothing_function=SmoothingFunction().method0)
    except ZeroDivisionError:
        bleu_1 = 0.0

    try:
        bleu_2 = sentence_bleu([t2_toks, ], t1_toks, smoothing_function=SmoothingFunction().method0)
    except ZeroDivisionError:
        bleu_2 = 0.0

    return bleu_1 + bleu_2

def symmetrical_WER(text_pair):

    t1, t2 = text_pair
    t1_lower = t1.lower()
    t2_lower = t2.lower()
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())

    try:
        wer_1 = edit_distance(t1_lower, t2_lower) / max((len(t1_toks), len(t2_toks)))
    except ZeroDivisionError:
        wer_1 = 0.0

    try:
        wer_2 = edit_distance(t2_lower, t1_lower) / min((len(t1_toks), len(t2_toks)))
    except ZeroDivisionError:
        wer_2 = 0.0

    return wer_1 + wer_2

def symmetrical_LCS(text_pair):

    t1, t2 = text_pair
    t1_lower = t1.lower()
    t2_lower = t2.lower()

    try:
        match1 = SequenceMatcher(None, t1_lower, t2_lower)
        M1 = match1.find_longest_match(0, len(t1_lower), 0, len(t2_lower))
        LCS_1 = M1.size

    except ZeroDivisionError:
        LCS_1 = 0.0

    try:
        match2 = SequenceMatcher(None, t2_lower, t1_lower)
        M2 = match2.find_longest_match(0, len(t2_lower), 0, len(t1_lower))
        LCS_2 = M2.size
    except ZeroDivisionError:
        LCS_2 = 0.0

    return LCS_1 + LCS_2

def symmetrical_ED(text_pair):

    t1, t2 = text_pair
    t1_lower = t1.lower()
    t2_lower = t2.lower()

    try:
        ED_1 = distance(t1_lower, t2_lower)
    except ZeroDivisionError:
        ED_1 = 0.0

    try:
        ED_2 = distance(t2_lower, t1_lower)
    except ZeroDivisionError:
        ED_2 = 0.0

    return ED_1 + ED_2


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

    # 01 Calculate NIST metric
    sample_text1 = texts
    sample_labels1 = labels
    sample_data1 = zip(sample_labels1, sample_text1)

    nist_scores = []
    for label, text_pair in sample_data1:
        nist_total = symmetrical_nist(text_pair)
        nist_scores.append(nist_total)
    print(nist_scores)
    corrnist, _ = pearsonr(nist_scores, sample_labels1)

    # 02 Calculate BLEU metric
    sample_text2 = texts
    sample_labels2 = labels
    sample_data2 = zip(sample_labels2, sample_text2)

    bleu_scores = []
    for label, text_pair in sample_data2:
        bleu_total = symmetrical_bleu(text_pair)
        bleu_scores.append(bleu_total)
    print(bleu_scores)
    corrbleu, _ = pearsonr(bleu_scores, sample_labels2)

    # 03 Calculate Word Error Rate metric
    sample_text3 = texts
    sample_labels3 = labels
    sample_data3 = zip(sample_labels3, sample_text3)

    wer_scores = []
    for label, text_pair in sample_data3:
        wer_total = symmetrical_WER(text_pair)
        wer_scores.append(wer_total)
    print(wer_scores)
    corrwer, _ = pearsonr(wer_scores, sample_labels3)

    # 04 Calculate Longest common substring metric
    sample_text4 = texts
    sample_labels4 = labels
    sample_data4 = zip(sample_labels4, sample_text4)

    lcs_scores = []
    for label, text_pair in sample_data4:
        lcs_total = symmetrical_LCS(text_pair)
        lcs_scores.append(lcs_total)
    print(lcs_scores)
    corrlcs, _ = pearsonr(lcs_scores, sample_labels4)



    # 05 Edit Distance metric
    sample_text5 = texts
    sample_labels5 = labels
    sample_data5 = zip(sample_labels5, sample_text5)

    ed_scores = []
    for label, text_pair in sample_data5:
        ed_total = symmetrical_ED(text_pair)
        ed_scores.append(ed_total)
    print(ed_scores)
    corred, _ = pearsonr(ed_scores, sample_labels5)


    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")

    # calculations are done in the previous steps, print out the results here
    print('Nist Pearsons correlation: %.3f' % corrnist)
    print('Bleu Pearsons correlation: %.3f' % corrbleu)
    print('WER Pearsons correlation: %.3f' % corrwer)
    print('LCS Pearsons correlation: %.3f' % corrlcs)
    print('ED Pearsons correlation: %.3f' % corred)


    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-train.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

