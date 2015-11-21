===Installing Biopython=== 

[[@http://biopython.org/wiki/Main_Page|Biopython]] is a great package if you're dealing with nucleotide or protein sequences. It offers support for many different types of files, including FASTA and FASTQ files.

The first step is to install the library. If you're using Windows, go to the <span style="background-color: #ffffff;">[[@http://biopython.org/wiki/Download|download]]</span> page and download the .exe file for Python 2.7.

I did not have an easy time installing Biopython on OSX. First, Canopy seems to require a subscription to their service in order to use their package installer, whichis named //enpkg//. That's no good - biopython is open source and free, there's no reason that one should have to pay for it!

That was the final (and only) nail in the coffin for me when it came to Canopy. I [[@https://support.enthought.com/entries/23580651-Uninstalling-Canopy|uninstalled]] it, and installed [[@http://continuum.io/downloads|Anaconda]]. Once Anaconda is installed, you can run the following command from your terminal to install Biopython:

[[code format="bash"]]
pip install biopython
[[code]]

To check if Biopython is installed, start up IPython and run the following import statement:
[[code format="python"]]
import Bio
[[code]]

If you get an error, then you do not have Biopython successfully installed. If the line runs without issue, then you're good to go!


===Learning and Using Biopython=== 

Once you have Biopython installed, you should start going through the [[@http://biopython.org/DIST/docs/tutorial/Tutorial.html|Biopython Tutorial]]. Here's a simple example adapted from the tutorial to read a FASTA fie:

[[code format="python"]]
import urllib2
from Bio import SeqIO

def download_to_file(url, output_file):
    """ Downloads a file from the internet, and writes it locally to an output file. """
    #open the URL
    response = urllib2.urlopen(url)

    #read the text from the URL
    text = response.read()

    #open a file handle to write to
    f = open(output_file, 'w')

    #write the file
    f.write(text)

    #close the file handle
    f.close()

def parse_fasta_file(file_url="http://biopython.org/DIST/docs/tutorial/examples/ls_orchid.fasta"):
    #download the file
    download_to_file(file_url, "ls_orchid.fasta")

    #iterate through each record in the FASFA file
    for seq_record in SeqIO.parse("ls_orchid.fasta", "fasta"):
        print("Sequence ID: {0}".format(seq_record.id))
        print("Sequence Length: {0} bases".format(len(seq_record)))
        print("First 30 characters of sequence: {0}".format(seq_record.seq[:30]))

if __name__ == "__main__":
    parse_fasta_file()

[[code]]

Placeholder...
