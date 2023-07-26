# Visual-Affordance-Model
Project for Computational Aspects of Robotics Course (COMSW4733) from Columbia University's School of Engineering and Applied Science, May 2023

When one designs learning algorithms for robots, how to represent a robot’s observation input and action outputs often plays a decisive role in the algorithm’s learning efficiency and generalization ability. In this project, I explored a specific type of action representation, Visual Affordance (also called Spatial Action Map): a state-of-the-art algorithm for visual robotic pack-and-place tasks. This method played a critical role for the MIT-Princeton team’s victory in the 2017 Amazon Robotics Challenge. This project uses PyBullet simulation engine extensively and the Visual Affordance model uses MiniUNet architecture.

First, I implemented and trained a Visual Affordance model with manually labeled data. Next, I implemented a method that further improves the performance of my model on unseen novel objects. Finally, I implemented Action Regression, an alternative action representation and explored its difference with Visual Affordance.

Packages Used:
- Miniforge
- Mambaforge

Visual Affordance:
Two key assumptions in this project:
1. The robot arm’s image observations come from a top-down camera, and the entire workspace is visible.
2. The robot performs only top-down grasping, where the pose of the gripper is reduced to 3 degrees of freedom (2D translation and 1D rotation).

Under these assumptions, we can easily align actions with image observations (hence the name spatial-action map). Visual Affordance is defined as a per-pixel value between 0 and 1 that represents whether the pixel (or the action directly mapped to this pixel) is graspable. Using this representation, the transnational degrees of freedom are naturally encoded in the 2D pixel coordinates. To encode the rotational degree of freedom, we rotate the image observation in the opposite direction of gripper rotation before passing it to the network, effectively simulating a wrist-mounted camera that rotates with the gripper.

Project Parts
1. Generate Training Data (pick labeler.py)
2. Implement Visual Affordance model (affordance model.py)
3. Improve Test-time Performance of Visual Affordance model
4. Alternative method: Action Regression (action regression model.py)
