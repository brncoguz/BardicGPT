# BardicGPT

This project implements a GPT-like language model from scratch using PyTorch. The model is trained on the Tiny Shakespeare dataset, which consists of text sequences from the works of Shakespeare. The goal is to generate text sequences in the style of Shakespeare using a transformer-based architecture.

# Example generated data

    MENENIUS:
    Good demesday, you before:
    You mean to another house, to enprove
    The saint-taste of his charity,
    As would becomfore a gentle-pater in the back.

    Servant:
    Pitain is it your sin By his haste; marrying
    yoed ather and squirbing for you whas she
    love-the meant have told your natural in the
    stains of an one four. Swear you were to virtuous
    as put locked into the fruit-bed moan'd with a meads;
    aidly, and these apped, ere all two kinsmen's liberance a glotten
    mads if for what it enemy, we will deh go'er, to content for
    the mortal letter, which sent bles, they were too soon.

    you can see more in the more.txt

#### Usage

##### Train the model by running:

    python3 gpt.py --mode train

###### Example Output

    step 0: train loss 4.2221, val loss 4.2306
    New best model saved with val loss 4.2306 at step 0
    step 500: train loss 1.7526, val loss 1.9108
    New best model saved with val loss 1.9108 at step 500
    step 1000: train loss 1.3913, val loss 1.5995
    New best model saved with val loss 1.5995 at step 1000
    ...
    step 4000: train loss 0.9568, val loss 1.5237
    step 4500: train loss 0.9057, val loss 1.5417
    step 4999: train loss 0.8537, val loss 1.5712
    Training complete. Best model saved at: checkpoints/best_model.pth with val loss 1.4881

#### --resume: Resume training from a previously saved checkpoint
    python3 gpt.py --mode train --resume

#### Generate Text
    python3 gpt.py --mode generate --context "To be or not to be" --max_new_tokens 100

###### Example Output

    Loaded model from the best model checkpoint.
    To be or not to bear yours.

    LUCIO:
    He's right.

    ISABELLA:
    Pray you, be your brother; he lmast, he did bumne a
    contrad

## Dataset

The model uses the Tiny Shakespeare dataset, which consists of sequences from Shakespeare's works.
