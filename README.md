# HATCH
This codebase contains implementation of paper Hierarchical Adaptive Contextual Bandits for Resource Constraint based Recommendation.

https://arxiv.org/abs/2004.01136

## Abstract
Contextual multi-armed bandit (MAB) achieves cutting-edge performance on a variety of problems. When it comes to real-world scenarios such as recommendation system and online advertising, however, it is essential to consider the resource consumption of exploration. In practice, there is typically non-zero cost associated with executing a recommendation (arm) in the environment, and hence, the policy should be learned with a fixed exploration cost constraint. It is challenging to learn a global optimal policy directly, since it is a NP-hard problem and significantly complicates the exploration and exploitation trade-off of bandit algorithms. Existing approaches focus on solving the problems by adopting the greedy policy which estimates the expected rewards and costs and uses a greedy selection based on each arm's expected reward/cost ratio using historical observation until the exploration resource is exhausted. However, existing methods are hard to extend to infinite time horizon, since the learning process will be terminated when there is no more resource. In this paper, we propose a hierarchical adaptive contextual bandit method (HATCH) to conduct the policy learning of contextual bandits with a budget constraint. HATCH adopts an adaptive method to allocate the exploration resource based on the remaining resource/time and the estimation of reward distribution among different user contexts. In addition, we utilize full of contextual feature information to find the best personalized recommendation. Finally, in order to prove the theoretical guarantee, we present a regret bound analysis and prove that HATCH achieves a regret bound as low as O(T‾‾√). The experimental results demonstrate the effectiveness and efficiency of the proposed method on both synthetic data sets and the real-world applications.

## Requirements:
python  3.7  
numpy   1.18.0  
pandas  1.0.3  
scipy   1.4.1  
sklearn 0.22.2  

## Files:
solver.ipynb   --the guidance of conducting HATCH  
bandit.py      --implement of HATCH  
evaluation.py  --the offline training and evaluation function  
utils.py       --the tools of main function  

## BIB
    
>@article{DBLP:journals/corr/abs-2004-01136,  
>&#160; &#160; &#160; &#160; author    = {Mengyue Yang and Qingyang Li and Zhiwei Qin and Jieping Ye},  
>&#160; &#160; &#160; &#160; ptitle     = {Hierarchical Adaptive Contextual Bandits for Resource Constraint based Recommendation},  
>&#160; &#160; &#160; &#160; journal   = {CoRR},  
>&#160; &#160; &#160; &#160; volume    = {abs/2004.01136},  
>&#160; &#160; &#160; &#160; year      = {2020},  
>&#160; &#160; &#160; &#160; url       = {https://arxiv.org/abs/2004.01136},  
>&#160; &#160; &#160; &#160; archivePrefix = {arXiv},  
>&#160; &#160; &#160; &#160; eprint    = {2004.01136},  
>&#160; &#160; &#160; &#160; timestamp = {Wed, 08 Apr 2020 17:08:25 +0200},  
>&#160; &#160; &#160; &#160; biburl    = {https://dblp.org/rec/journals/corr/abs-2004-01136.bib},  
>&#160; &#160; &#160; &#160; bibsource = {dblp computer science bibliography, https://dblp.org}  
>}

