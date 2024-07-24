import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.unicode import *
from utils.times import *
from models.GRU import *
from models.LSTM import *
from models.BiLSTM import *
from models.attention import *
from models.attention2 import *
from models.transformer import *
from sklearn.model_selection import train_test_split
import fire
import json
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

rouge = ROUGEScore()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5

def train(model, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    if model == "dot" or model == "bilinear" or model == "multi_layer_perceptron" or model == 'QK':
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_hiddens = encoder_hidden
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
            encoder_hiddens = torch.cat((encoder_hiddens, encoder_hidden), dim=0)
    elif model != "transformer":
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
    else:
        encoder_outputs = encoder(input_tensor)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    if model == "transformer":
        decoder_hidden = encoder_outputs.unsqueeze(0)
    elif model == "BiLSTM":
        decoder_hidden = encoder_hidden
        decoder_hidden = torch.mean(decoder_hidden[0], dim=0).unsqueeze(0)
    else:
        decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):

            # attention works on the encoder hidden states and the decoder hidden state
            if model == "dot" or model == "bilinear" or model == "multi_layer_perceptron" or model == 'QK':
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_hiddens)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)


            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if model == "dot" or model == "bilinear" or model == "multi_layer_perceptron" or model == 'QK':
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_hiddens)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)


            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(model, input_lang, output_lang, train_pairs, encoder, decoder, epochs, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    iter = 1
    n_iters = len(train_pairs) * epochs

    for epoch in range(epochs):
        print("Epoch: %d/%d" % (epoch, epochs))
        pbar = tqdm(train_pairs, total=len(train_pairs))
        for training_pair in pbar:
            training_pair = tensorsFromPair(training_pair, input_lang, output_lang)

            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(model, input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            print_loss_avg = print_loss_total
            print_loss_total = 0
            pbar.set_description('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            iter +=1
            plot_losses.append(print_loss_avg)
    return plot_losses


def evaluate(model, input_lang, output_lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]

        if model == "dot" or model == "bilinear" or model == "multi_layer_perceptron" or model == 'QK':
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
            encoder_hiddens = encoder_hidden
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
                encoder_hiddens = torch.cat((encoder_hiddens, encoder_hidden), dim=0)
        elif model != "transformer":
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
        else:
            encoder_outputs = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        if model == "transformer":
            decoder_hidden = encoder_outputs.unsqueeze(0)
        elif model == "BiLSTM":
            decoder_hidden = encoder_hidden
            decoder_hidden = torch.mean(decoder_hidden[0], dim=0).unsqueeze(0)
        else:
            decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            if model == "dot" or model == "bilinear" or model == "multi_layer_perceptron" or model == 'QK': 
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_hiddens)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# def evaluateRandomly(encoder, decoder, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words = evaluate(encoder, decoder, pair[0])
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')


def test(model, input_lang, output_lang, encoder, decoder, testing_pairs):
    input = []
    gt = []
    predict = []
    metric_score = {
        "rouge1_fmeasure":[],
        "rouge1_precision":[],
        "rouge1_recall":[],
        "rouge2_fmeasure":[],
        "rouge2_precision":[],
        "rouge2_recall":[]
    }
    from tqdm import tqdm
    for i in tqdm(range(len(testing_pairs))):
        pair = testing_pairs[i]
        output_words = evaluate(model, input_lang, output_lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        input.append(pair[0])
        gt.append(pair[1])
        predict.append(output_sentence)

        try:
            rs = rouge(output_sentence, pair[1])
        except:
            continue
        metric_score["rouge1_fmeasure"].append(rs['rouge1_fmeasure'])
        metric_score["rouge1_precision"].append(rs['rouge1_precision'])
        metric_score["rouge1_recall"].append(rs['rouge1_recall'])
        metric_score["rouge2_fmeasure"].append(rs['rouge2_fmeasure'])
        metric_score["rouge2_precision"].append(rs['rouge2_precision'])
        metric_score["rouge2_recall"].append(rs['rouge2_recall'])

    metric_score["rouge1_fmeasure"] = np.array(metric_score["rouge1_fmeasure"]).mean()
    metric_score["rouge1_precision"] = np.array(metric_score["rouge1_precision"]).mean()
    metric_score["rouge1_recall"] = np.array(metric_score["rouge1_recall"]).mean()
    metric_score["rouge2_fmeasure"] = np.array(metric_score["rouge2_fmeasure"]).mean()
    metric_score["rouge2_precision"] = np.array(metric_score["rouge2_precision"]).mean()
    metric_score["rouge2_recall"] = np.array(metric_score["rouge2_recall"]).mean()

    print("=== Evaluation score - Rouge score ===")
    print("Rouge1 fmeasure:\t",metric_score["rouge1_fmeasure"])
    print("Rouge1 precision:\t",metric_score["rouge1_precision"])
    print("Rouge1 recall:  \t",metric_score["rouge1_recall"])
    print("Rouge2 fmeasure:\t",metric_score["rouge2_fmeasure"])
    print("Rouge2 precision:\t",metric_score["rouge2_precision"])
    print("Rouge2 recall:  \t",metric_score["rouge2_recall"])
    print("=====================================")
    return input,gt,predict,metric_score

def main(model='GRU'):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    X = [i[0] for i in pairs]
    y = [i[1] for i in pairs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    train_pairs = list(zip(X_train,y_train))
    test_pairs = list(zip(X_test,y_test))


    hidden_size = 512
    epochs = 10
    learning_rate = 0.01

    if model == 'GRU':
        encoder = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
        decoder = Decoder(hidden_size, output_lang.n_words, device).to(device)
    elif model == 'LSTM':
        encoder = EncoderLSTM(input_lang.n_words, hidden_size, device).to(device)
        decoder = DecoderLSTM(hidden_size, output_lang.n_words, device).to(device)
    elif model == 'BiLSTM':
        encoder = EncoderBiLSTM(input_lang.n_words, hidden_size, device).to(device)
        decoder = Decoder(hidden_size, output_lang.n_words, device).to(device)
    elif model == 'dot' or model == 'bilinear' or model == 'multi_layer_perceptron' or model == 'QK':
        encoder = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
        decoder = AttentionDecoder(hidden_size, output_lang.n_words, device, model).to(device)
        # decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, device).to(device)
    elif model == 'transformer':
        encoder = EncoderTransformer(input_lang.n_words, hidden_size, device).to(device)
        decoder = Decoder(hidden_size, output_lang.n_words, device).to(device)

    losses = trainIters(model, input_lang, output_lang, train_pairs, encoder, decoder, epochs, learning_rate)
    # write losses to npz file
    input,gt,predict,score = test(model, input_lang, output_lang, encoder, decoder, train_pairs)
    input,gt,predict,score = test(model, input_lang, output_lang, encoder, decoder, test_pairs)

    np.savez(f"results/{model}.npz", losses=losses, score=score)

if __name__ == "__main__":
    fire.Fire(main)