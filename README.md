# Deep Mutual Learning Framework

This is a [Hackathon 2021 project](https://garagehackbox.azurewebsites.net/hackathons/2356/projects/107707)

We are building a Python library for Deep Mutual Learning and apply it to compress and
optimize some of the models we have in production. We use this paper as an inspiration:
[Deep Mutual
Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf)

The idea is quite simple: we run the usual {train - apply structured pruning - repeat} loop, but
train multiple clones of the same model in parallel. All clones initially have the same
architecture, but different random initialization; when we train them, to the existing loss
function we also add a consensus loss (implemented in our library) that pulls all models closer
together. The paper shows that this approach outperforms other traditional approaches to model
pruning. 

*If you already **have a working model** and especially of you've tried to apply iterative
**structured pruning** to it - come work with us!* With a few extra lines of code you may be able
to train a much better and compact version of your neural net.
