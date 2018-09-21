#KerasAdversary#

This library provides several simple but powerful interfaces on adversarial attacks.

##Dependencies##

This library is mainly designed to attack the target models built with **Keras**. However, RNN is not supported now.

To use our library in your project, several libraries are pre-requisites:

- **Tensorflow** (tested on 1.2.0)
- **keras** (tested on 2.0.8)
- **scipy** (to support visual functions in our library)
- **matplotlib** (to support visual functions in our library)
- **numpy** (to support matrix calculation)

##Introductions##

All the files can be found in the folder *lib/*.

###AdversarialAttacks.py###

This file mainly contains several classes for adversarial attacks.

1. **DLWhiteBoxAttacks**: FGSM, BIM, C&W
3. **DLBlackBoxAttacks**: TODO
2. **MLWhiteBoxAttacks**: TODO
 
Now all the attack algorithms are for target attacks.

Adversarial attack can be quite simple with our library. For example, if *model* is the target model built with Keras, then it can be attacked by FGSM with the following commands:
	
	from lib.AdversarialAttacks import DLWhiteBoxAttacks
	import keras.backen as K	
	AttackAgent = DLWhiteBoxAttacks(model, K.get_session())
	adv_x = AttackAgent.fgsm(x, y, epsilon=0.1, clip_min=0., clip_max=1.)

where *adv_x* are adversarial examples of x.

###visualization.py###

This file contains several methods for visualizations.

###utils.py###

This file contains several useful methods that you may use in your project.