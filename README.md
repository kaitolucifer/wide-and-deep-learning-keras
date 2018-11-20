# Wide and Deep Learning implemented with Keras
<pre>
            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                    Version 2, December 2004

 Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. You just DO WHAT THE FUCK YOU WANT TO.
</pre>

### Requirements
* Python >= 3.6
* TensorFlow >= 1.6
* Keras >= 2.0.0

### Model Plot
![Model Plot](https://github.com/kaitolucifer/wide-and-deep-learning-keras/blob/master/model.png)
The model is based on [Heng-Tze Cheng, *et al.* Wide & Deep Learning for Recommender Systems (2016)](https://arxiv.org/abs/1606.07792)<br/>
I used [UCI Machine Learning Repository: Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult) as example data.</br>
There are 8 categorical features so I put every one of them into a embedding layer.</br>
And I just put the rest 5 continuous feature into a dense layer and concatenate it with all embedding layers.</br>
Then add some dense layer and before sigmoid layer I concatente the dense layer output with logistic part input.</br>

### Best Test Set Accuracy
85.8%
