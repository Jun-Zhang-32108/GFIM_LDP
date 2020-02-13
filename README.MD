This is the bachelor thesis work of Jun Zhang done in [State Key Laboratory of Networking and Switching Technology](https://sklnst.bupt.edu.cn), supervised by Xiang Cheng.

Abstract 
======================================
With the development of information technology, popularity of smart devices and the
extensive uses of sensors, the data volume in the world presents explosively increasing.
Nowadays, our society has entered the era of big data. Set-valued data is one kind of classical
big data. By mining the frequent items of users’ set-valued data, data aggregator can learn
about the preferences of users, which can provide support for decisions. However, set-valued
data contains a great number of sensitive information of users, directly reporting the frequent
items and the corresponding counts or frequencies could lead to the leakage of users’ privacy.
As the state-of-the-art privacy protection model, local differential privacy(LDP) provides a
feasible solution for such problem.

In this paper, we propose a frequent items mining algorithm with ε-LDP, named Groupbased Frequent Items Mining (GFIM). The main idea of this algorithm is to first split the
users randomly into two groups with the same size. Basing on the user data of the first group
we gather the candidate set of possible frequent items, then we refine the candidate set using
the user data of the second group. Finally, we obtain the estimated frequent items and the
corresponding frequencies by combining the results of these two phases. This paper
theoretically proves that GFIM satisfies ε-LDP. In addition, extensive experiments on
synthetic dataset and real dataset demonstrate its effectiveness and superiority over existing
method.


The datasets I used in the experiments can be found [here](https://drive.google.com/open?id=1RMOij0O_aMXnOu4RtGGsr-ERH1Q8RjXQ).
