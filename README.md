# KerasAdversary #

This library provides several simple but powerful interfaces on adversarial attacks.

## Dependencies ##

This library is mainly designed to attack the target models built with **Keras**. However, RNN is not supported now.

To use our library in your project, several libraries are pre-requisites:

- **Tensorflow** (tested on 1.2.0)
- **keras** (tested on 2.0.8)
- **scipy** (to support visual functions in our library)
- **matplotlib** (to support visual functions in our library)
- **numpy** (to support matrix calculation)

## Introductions ##

All the files can be found in the folder *lib/*.

### KerasAdversary.py ###

This file mainly contains several classes for adversarial attacks.

1. **WhiteBoxAttacks**: L-BFGS-B, FGSM, BIM, C&W(TODO)
2. **BlackBoxAttacks**: TODO

Now it supports both target attacks and non-target attacks.

Adversarial attack can be quite simple with our library. For example, if *model* is the target model built with Keras, then it can be attacked by FGSM with the following commands:
	
	from lib.AdversarialAttacks import WhiteBoxAttacks
	import keras.backen as K	
	AttackAgent = WhiteBoxAttacks(model, K.get_session())
	adv_x = AttackAgent.fgsm(x, y, epsilon=0.1, clip_min=0., clip_max=1.)

where *adv_x* are adversarial examples of x.

**NOTE**: If you want to perform **target attacks**, then you should set the parameter **target=True** and **y is your target label**. For example,

	from lib.AdversarialAttacks import WhiteBoxAttacks
	import keras.backen as K	
	AttackAgent = WhiteBoxAttacks(model, K.get_session())
	adv_x = AttackAgent.fgsm(x, y, target=True, epsilon=0.1, clip_min=0., clip_max=1.)

### visualization.py ###

This file contains several methods for visualizations.

Here are some examples:

1. **mnist**

	![](https://github.com/ZhangXiao96/KerasAdversary/blob/master/pic/mnist_adv.jpg)

2. **cifar10**

	![](https://github.com/ZhangXiao96/KerasAdversary/blob/master/pic/cifar10_adv.jpg)

### utils.py ###

This file contains several useful methods that you may use in your project.
