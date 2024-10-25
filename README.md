# wildlife

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
If they have changes, do those changes on your branch. If it is perfect you can merge to main. Click ‘squash and merge’ on the PR.

### 10 Delete your upstream branch on gitHub.

### 11 Repeat for your next feature! 
Starting at step 1.
