import keras
from readData import tokenizer, pad_sequences, max_length, email_type

model = keras.models.load_model("F:/FraudDetection/model/model.h5")

def predictFraud(email_input:str):
    tokenized_input = tokenizer.texts_to_sequences(email_input)
    padded_words = pad_sequences(tokenized_input, maxlen=max_length)
    prediction = model.predict(padded_words)[0]
    thres = 0.3 # Change recall threshold
    print(prediction)
    predicted_class = [email_type[0] if prediction > thres else email_type[2]]
    return predicted_class

if __name__ == "__main__":
    text_input = """

Subject: Exclusive Offer: Get Rich Quick! 💰💸 Act Now!

Dear Valued Subscriber,

Congratulations! You've been selected for an incredible opportunity that could change your life forever. We are excited to present to you the "Get Rich Quick" program, a revolutionary system that guarantees massive earnings with minimal effort. 💯

Imagine living the life you've always dreamed of—luxurious vacations, extravagant shopping sprees, and financial freedom. This is your chance to make those dreams a reality. Thousands of individuals just like you have already embarked on this journey to financial success. Now, it's your turn!

Here's how it works:

Register for the "Get Rich Quick" program for a one-time fee of only $99.99.
Gain access to our exclusive online platform, which leverages cutting-edge artificial intelligence algorithms to maximize your profits.
Sit back and watch as the money starts pouring into your account. No special skills or prior experience required!
But wait, there's more! As a special bonus for taking immediate action, we'll also provide you with a top-secret "Insider Trading" guide that will give you an unfair advantage in the stock market. 💹

Act now, and you'll receive an additional surprise gift—an all-expenses-paid trip to a tropical paradise! 🏝️

Hesitate no longer! Seize this once-in-a-lifetime opportunity before it's too late. Our exclusive offer won't last forever, and spots are filling up quickly. Don't let this chance slip through your fingers!

WARNING: This opportunity is limited to a select few, and once the slots are filled, we won't be able to accept any more participants. Don't miss out on this golden ticket to financial abundance.

Please remember to keep this opportunity confidential, as we want to ensure only serious individuals join our program. We strictly adhere to a privacy policy to safeguard your personal information.

To secure your spot in the "Get Rich Quick" program, click the link below:

[Malicious Link: Do not click this link; it is for illustrative purposes only]

Thank you for your time, and we look forward to welcoming you to the ranks of the wealthy elite!

Best regards,

The "Get Rich Quick" Team

Note: This email is for testing purposes only and does not represent a legitimate offer or opportunity. Please exercise caution and avoid clicking on any suspicious links in real-life situations.






"""
    predicted_words = predictFraud(text_input)
    print(predicted_words)