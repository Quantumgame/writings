=Topics= 
# The Terminal
# IPython
# Creating and Running Scripts from the Terminal
# Console I/O
# Conditional Statements
# Functions
# Loops

=In-Class Projects= 


===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">1. Introduction to the Terminal</span>=== 

Mac Users: [[@http://mac.appstorm.net/how-to/utilities-how-to/how-to-use-terminal-the-basics/|Check out this tutorial]].
Linux Users: [[@http://linuxcommand.org/lc3_learning_the_shell.php|Check out this tutorial]].
Windows Users: Install [[@https://github.com/msysgit/msysgit/releases/|GitBash]]

The terminal is a powerful way of interacting with your computer, you can do stuff like:
# List, copy, rename and move files and directories
# Search the filesystem for a given file
# Search for text within files
# Execute arbitrary commands

Have you ever used [[@http://en.wikipedia.org/wiki/Secure_Shell|ssh]]? In this world of cloud computing, there will be times when you do not have access to a graphical interface to a server. The use of the terminal will be essential in effectively administering your remote computer! Let's get into the essential operations:

**pwd: figuring out where you are**
The //pwd// command prints out the directory that you are currently in. It stands for "print working directory". The "working directory" is the current directory you are in. Open your terminal, and type "pwd". That's where you are. You are probably starting out in your //home// directory.

**ls: listing the contents of the current directory**
Type "ls" and you should get a list of several files. There are two special files, "." and "..". The single dot represents the directory you are in, while the double dot represents the parent directory of the directory you are in. Maybe "ls" stands for "list", I don't know. Doesn't matter!

**cd: changing to a new directory**
"cd" stands for "change directory". Do you want to go home? The tilde "~" is a special keyword reserved for your home directory. If you ever get lost at the terminal, don't freak out, you can go home by typing:
[[code format="bash"]]
cd ~
[[code]]
If you want to go to the parent directory, type this:
[[code format="bash"]]
cd ..
[[code]]

**cp: copying files**
You can copy one file to another using the "cp" command:
[[code format="bash"]]
cp <file location> <new location>
[[code]]

**rm: removing files**
I AM NOT RESPONSIBLE FOR DAMAGE TO YOUR COMPUTER FROM THIS CLASS! rm stands for "remove", it's dangerous. The files you delete go away forever, to binary hell, you cannot retrieve them. Consider deleting things from the finder on mac or explorer on Windows. If you're on Linux, I imagine you've probably screwed up your computer so many times that you're quite comfortable using rm at this point, given all that you have suffered.

**mkdir/rmdir: create or remove a directory**
You can create a directory using mkdir, and remove an *empty* directory using rmdir. If you want to remove a directory and all it's contents, do this BUT IT'S NOT MY FAULT IF YOU DELETE YOUR HARD-EARNED RESEARCH:
[[code format="bash"]]
rm -fr <directory with stuff in it>
[[code]]

**more/less/cat: list a file's contents**
Any one of these commands will list a file's contents. "cat" just spits out everything to the Terminal, while "more" and "less" are polite and only show a little bit at a time.

**top: show the running processes**
This command is cool. It shows you all the programs that are running on your computer. The most relevant columns it displays are the "pid", which is the //process ID// of the program that is running, and %CPU and %MEM, which show the amount of CPU and memory that program takes up.

**kill: stop a running process**
Once you have the process ID of a program that is running, you can kill it using the following command:
[[code format="bash"]]
kill -9 <process id>
[[code]]
Killing processes is awesome, but many processes that you see when running top are vital to your computer functioning properly. Only kill processes that you know are bad.

**env: list the environment variables**
Environment variables are system-wide variables that can be used by any application. To list all the environment variables on your machine, type the following in a terminal window:
[[code format="bash"]]
env
[[code]]

**export: set an environment variable**
If you want to add a new directory to your PATH or PYTHONPATH variables, you can do it like this:
[[code format="bash"]]
export PATH=$PATH:/the/new/directory
[[code]]
When you reference environment variables, you have to put a dollar sign in front of them. If you want to print the value of an environment variable out to the screen, you can use the "echo" command:
[[code format="bash"]]
echo $PATH
[[code]]

There are two important environment variables we'll work with: PATH and PTYHONPATH. The PATH variable is a colon-separated (semi-colon on Windows) list of directories that contain executable files. For example, the command "ls" is usually in the directory /usr/bin, but you can run it from any directory. The PYTHONPATH variable is a colon-separated list of directories to that the Python interpreter uses to find source code.

**EXERCISE:** Use a combination of "ls" and "cd" to navigate through your filesystem. If you're on a Mac, check out the "/Applications" directory. If you're on Windows, navigate using cygwin to the "/cygwin/c" directory. If you're using Linux, check out "/" and "/usr". On Mac or Linux, check out the "/tmp" directory (on Windows, should be /cygdrive/c/temp). This directory is wiped clean every time you restart your computer. Go to the temp directory, create some directories, delete them, do what you please.

**EXERCISE:** Create a folder to keep your scripts and files for this class. Navigate to your Documents folder (or wherever you would like the files to reside) and use mkdir to create a new folder.



===2. Introduction to IPython=== 

IPython is a //command line interface// (CLI) that //interprets// Python code. You can start it at your terminal by typing:
[[code format="bash"]]
ipython --pylab
[[code]]
<span style="line-height: 1.5;">Did you just get an error message? There are a handful of situations in which the Enthought distribution does not successfully work, and you have just encountered one of them. Please contact us immediately and we'll try our best to work it out. If you're on Mac, and get an error that says "ipython not found", try pasting this into your terminal:</span>
[[code format="bash"]]
export PATH=~/Library/Enthought/Canopy_64bit/User/bin:$PATH
[[code]]

We used IPython in the first class - it was embedded in the Canopy application. Now we are using IPython in it's native format. The "--pylab" argument allows IPython to operate in a way that makes it easy to interactively create plots and analyze data using [[@http://www.numpy.org|NumPy]], [[@http://www.scipy.org|SciPy]], and [[@http://www.matplotlib.org|matplotlib]]. Do you need some proof? Execute this line of code in IPython:
[[code format="python"]]
imshow(randn(100, 300))

[[code]]
Congratulations! You just generated a Gaussian random matrix and plotted it. There will be much more where that came from in future lectures.

IPython is a very worthwhile tool when playing around with code and analyzing data. Check out the <span style="background-color: #ffffff;">[[@http://ipython.org/ipython-doc/2/interactive/tutorial.html|IPython tutorial]]</span> and become a power user!
But when it comes to actually putting together programs, we need to aggregate our code into files and run them from the Terminal.
