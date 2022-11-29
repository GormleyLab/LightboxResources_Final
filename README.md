# LightboxResources_Final

This repository contains all files necessary for building a lightbox with 96 multiplexed LEDS. To use any Python/Arduino scripts, ensure that the COM ports and fluorescence data file paths are changed to your specific setup. 

-STL Files are available for 3D printing a shell to contain all electronic components on the lightbox.

-Lightbox schematic shows the circuit diagram in which all components should be wired. 

-Python Files are used for fluorescence data analysis and sending data to the lightbox for lighting.
  - LightboxCalculator&Tester.py
    - tkinter interface that allows one to calculate the binary number to control each row of the lightbox. Click the LED in a row to be turned on, then click "Total". After doing so for each row, the final array should be input as "<a,b,c,d,r,f,g,h>" in the Python Script.
    
  - LBDataComm_NoFluorescence.py
    - Control the lighting profile of the lightbox using a manually input data array. Use the "LightboxCalculator&Tester.py" to determine the input array.
    
  - LBDataTest_FluorescenceXML.py
    - Use fluorescence data to control which LED is turned on. Two sets of sample data have been provided. Edit the clean_data() function and pass the file path of a given sample file.

  - FluorescenceAnalysisContinuous.ipynb
    - A jupyter notebook that can be used to continuously analyze sample data files in a given experiment. After running cells 1-3, cell 4 can be run over and over again with the sample file path name changed in order to simulate an automated fluorescence experiment. Of the sample data given, each experiment contains five timepoints with eight wells containing fluorescence data. So for example, run clean_data(r'C://...../SampleData1_1), then clean_data(r'C://...../SampleData1_2), then clean_data(r'C://...../SampleData1_3), etc.

-Arduino .ino files are uploaded to the Ardruino module used in the lightbox for PC communicationa and LED multiplexing.

-Lightbox PCB Files folder contains all output files for third-party manufacturers.
