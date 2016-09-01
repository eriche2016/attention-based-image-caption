# attention-based-image-caption

code for [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)
runnable on GPU and CPU.

currently support soft attention version

## Dependencies

This code is written in lua. To use it you will need:
* torch7
* 
* ...
Basically same dependencies as [neuraltalk2](https://github.com/karpathy/neuraltalk2/)


To use the evaluation script (metrics.py): see
[coco-caption](https://github.com/tylin/coco-caption) for the requirements.

## Reference
### This repositories roughly builds the soft attention based model  on top of [neuraltalk2](https://github.com/karpathy/neuraltalk2/).
If you use this code as part of any published research, please acknowledge the
following paper (it encourages researchers who publish their code!):

**"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."**  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan
Salakhutdinov, Richard Zemel, Yoshua Bengio. *appeared ICML (2015)*

    @article{Xu2015show,
        title={Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
        author={Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1502.03044},
        year={2015}
    } 

## License

The code is released under a [revised (3-clause) BSD License](http://directory.fsf.org/wiki/License:BSD_3Clause).


