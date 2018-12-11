# xss-traceback

Using machine leanrning method to traceback xss attacks.

- `data/train.csv` 为 `url` 编码之后的向量，每一行代表一个url，长度为100

- `data/labels.csv` 为 `Author` encode之后的数据，数值的意思为one-hot向量为1的index. 

- `data/labels_map.csv` 为 `Author` 的类别(结合 `data/labels.csv` )，可根据此文件还原作者信息。如文件中的第10行为 `04hrb` ， 那么预测结果为index=10的话，就是说预测结果为该作者。

	训练的时候，如果是用softmax作为分类器的话，需要将index转为one-hot向量作为ground_truth_y