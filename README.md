### How I scored the words?  
To score the words I used `TF-IDF`, for `IDF` we need some bunch of documents,
but we only have one document.  
But inside that document we have multiple section, so I treated those sections
as Documents, so now we have multiple Documents to apply `TF-IDF`.  

### How I calculated sentence score

So after we got `TF-IDF` scores for each word, we can see how those scores are distributed.  
Now we can get the Standard deviation `(spread)` of these scores.  
Now to weight each sentence, we take the `TF-IDF` score of each word in that sentence, 
and apply this formula.  
`weight[word] = p_U * STD + TF-IDF[word]`   

- p_U: proportion of uppercase letters in the word.
- STD:  Standard deviation of TF-IDF scores

Now we take relevant word, how we define relevance is, 
now that we get all the weights for our sentence, 
we take the weights who are above the `1st standard deviation` of the `TF-IDF` scores, 
and take average of these weights and this is the weight of our sentence.  


### How am I ensuring that two consecutive sentences are not more than D sentences apart.

Now as we're putting some rules to weight the sentence, so some 
sentences will be ruled out from weight calculation, 
so we have weight only for relevant sentences, 
so it might happen that there are more than `D` consecutive sentences aren't weighted.

So to deal with it I divided sentences into clubs (partitions) where each club follow the
property of cosecutive sentences with max distance of `D`, and use the sliding window approach
on those clubs.

### Further scope of improvement.

Making it more efficient, reducing the Time and Space Complexity.   
Extracting much reliable summary out of the Document.
