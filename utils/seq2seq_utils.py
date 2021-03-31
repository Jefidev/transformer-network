import spacy
import torch
from torchtext.data.metrics import bleu_score


def translate_sentence(
    model, sentence, voc_english, voc_french, device, max_length=300
):
    spacy_french = spacy.load("fr_core_news_sm")

    tokens = [token.text for token in spacy_french.tokenizer(sentence)]
    french_transform = lambda x: [voc_french["<sos>"]] + [
        voc_french[token] for token in x + [voc_french["<eos>"]]
    ]

    text_to_indices = french_transform(tokens)
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.Encoder_LSTM(sentence_tensor)

    outputs = [voc_english["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == voc_english["<eos>"]:
            break

    translated_sentence = [voc_english[idx] for idx in outputs]
    return translated_sentence[1:]


def bleu(test_iter, model, voc_english, voc_french, device):
    targets = []
    outputs = []
    spacy_english = spacy.load("en_core_web_sm")

    for example in test_iter:
        src = example[0]
        trg = example[1]

        prediction = translate_sentence(model, src, voc_english, voc_french, device)
        prediction = prediction[1:-1]  # remove <sos> and <eos> token
        prediction = [voc_english.itos[token] for token in prediction]

        # Tokenize target :
        tokens = [token.text for token in spacy_english.tokenizer(trg)]
        english_transforms = lambda x: [voc_english[token] for token in x]
        # Transforming to list of idx
        trg = english_transforms(tokens)
        trg = [voc_english.itos[token] for token in trg]

        targets.append(trg)
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss):
    print("saving")
    print()
    state = {
        "model": model,
        "best_loss": best_loss,
        "epoch": epoch,
        "rng_state": torch.get_rng_state(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, "./models/checkpoint-seq2seq")
    torch.save(model.state_dict(), "./models/checkpoint-seq2seq-SD")

