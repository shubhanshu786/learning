# obfuscated-multiclassification

**obfuscated multiclassification** is a classification problem with character sequences and associated labels with them. Data is available at **https://www.kaggle.com/alaeddineayadi/obfuscated-multiclassification**

In this task i am using/learning, how to use CNN with text data. Here idea is, through the slides of CNN window (kernal) it learn/capture different important feature from data. In case of text data, CNN try to capture **local/temporal dependencies** in data in a certain range(N-grams in case of data-provided by kernal-size). For example, in case of kernal_size = 5, model will try to capture important features (like certain patterns present in data) in 5-grams(character in this case) of data. This way, model with different kernal_size can capture important feature in different character ranges, which could be use for classification purpose.

Here is the complete flow:
--

1) Encode all characters in one hot encoding (since only 26 characters so hot encoding size will be 26)

2) Convert all sentences into same length (452 in this data) with the help of 0 padding at the end.

3) Input same generated data to all 8 different CNN layers with different kernal_size. (here kernal size will be **choosen_character_sequence_length x feature_vector_size**) Here feature_vector_size is the most important part because we wanted to consider whole character during training time, rather than just a part of it. Do remember, it's text data not image one. :-P

4) Once all CNN provide data after maxpooling and flatten layer, merge all 8 layers data to use them for classification purpose.

5) Use dense leyers to build classification model.

## Results
### Model Accuracy
![](https://github.com/shubhanshu786/learning/blob/master/obfuscated-multiclassification/Training%20Accuracy.png "Model Accuracy")
### Model Loss
![](https://github.com/shubhanshu786/learning/blob/master/obfuscated-multiclassification/Training%20Loss.png "Model Loss")
### Validation Accuracy
![](https://github.com/shubhanshu786/learning/blob/master/obfuscated-multiclassification/Validation%20Accuracy.png "Validation Accuracy")
### Validation Loss
![](https://github.com/shubhanshu786/learning/blob/master/obfuscated-multiclassification/Validation%20Loss.png "Validation Loss")

Blue : CNN Model with 64 features

Red : CNN Model with 128 features
## Note
1) Selection of number features (**128** in my case) and number of different CNN models is selected based on best results.

2) Graph smoothness represent, how powerful CNN models are, with text data classification.

For any query, and clearance drop me mail at **dec31.shubh@gmail.com**
