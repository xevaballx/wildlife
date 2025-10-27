Abstract: Camera traps have revolutionized wildlife conservation research by generating vast numbers of images, but their imbalanced species distributions pose a significant challenge to automatic classification. We apply cost-sensitive loss functions and various preprocessing methods to address these issues, significantly improving performance.


According to a recent report from the World Wildlife Fund (WWF), the average wildlife population has decreased by 73% over the past 50 years. This reduction in biodiversity is driven by various factors, including habitat destruction, pollution, and over-exploitation of species. Biodiversity is not only a measure of ecological health but also a cornerstone of human well-being, influencing critical aspects such as air and water quality, food security, and the dynamics of zoonotic diseases.

Advancements in machine learning (ML) are instrumental in addressing environmental challenges by enabling more effective monitoring and analysis of ecosystems. One area of growing interest is the application of ML to automatically classify camera trap images, which are widely used in ecological studies to estimate species abundance and activity patterns. These sensor-triggered cameras generate vast amounts of images and video that are labor intensive and costly (if not impossible) to label manually.

Our work focuses on addressing the critical challenge of class imbalance in camera trap datasets. We test these methods across two ML architectures: ResNet50 and EfficientNetV2. While CNNs have shown promise in this field, camera trap data presents unique challenges: images often capture only partial views of animals,  exhibit high intra-class variability due to diverse backgrounds, vary in quality and light, and suffer from significant class imbalances that mirror the natural distribution of ecosystems. Such imbalances can bias models toward abundant species, limiting their utility for ecological studies. Achieving balanced performance across species is vital, particularly for studying rare or elusive animals that hold significant ecological importance. As such, solutions are required to address this imbalance.

Thusly, we explore the impact of various loss functions and data augmentation strategies to address class imbalance in camera trap datasets. To the best of our knowledge, this is one of the first studies to systematically evaluate these techniques across multiple architectures for this application. Our experiments demonstrate that these approaches improve performance on minority classes with minimal impact on majority class performance, leading to a notable increase in macro F1-score.

---

# github workflow

Note: This doc is based on using the command line (terminal) in VSCode, not the VSCode extension for git.

### 0 Clone the repo
Go to GitHub in your browser. Go to our team repo and clone the project.
Click the green “code” button and clone the repo however you like. 
Copy the link.
On the command line type:
`git clone <paste link>`

### 0.1 Get your bearings
cd into the folder created by cloning, then type:
`ls` to see all the files

### 1 Ensure you are on the Main Branch
`git checkout main` 

### 2 Pull any new changes from Main Branch
Ensure you have the most up-to-date code
`git pull`

### 3 Create your branch
`git branch <YourFeatureHere>`
example: `git branch DogDoorActivator`

### 4 Checkout your branch
`git checkout <YourFeatureHere>`
example: `git checkout DogDoorActivator`

### 4.1 Make sure you are where you want to be
You can always type:
`git branch` to make sure you are on the branch you want to be on

### 5 Set the remote as upstream so you can eventually push changes to GitHub
`git push --set-upstream origin DogDoorActivator`

### 6 Write your code on this branch.

### 7 “Commit early push often.”
We want to push our commits to the GitHub remote feature branch often. When you are done for the day or want to take a break, commit and push.
From the root of the project folder do all three of the following:
1. `git add .` this tracks all of the files in this directory and its subdirectories 
2. `git commit -m “<YourCommitMessageHere>” `
3. `git push` this pushes any unsynced commits to remote branch on gitHub

### 7.1 Document/Comment and Test
Comment and test your code before you ask for a PR so the reviewer will have an easier time.

### 8 Pull Request (PR)
When you are completely done with your feature and you are ready to merge these changes to main, go to the GitHub repo and open a Pull Request (PR) from your branch to Main.

### 9 Ask one of your peers to review. 
If the peer has changes for you, do those changes on your branch. If it is perfect, they can merge it for you or approve it for you to merge. Click ‘squash and merge’ on the PR.

### 10 Delete your upstream branch on gitHub.
Or depending on your workflow, you can keep using your branch.

### 11 Repeat for your next feature! 
Starting at step 1.
