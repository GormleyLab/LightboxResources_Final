#THIS VERSION IS USED TO CALCULATE DATA TO SEND TO THE LIGHTBOX USING FLUORESCENCE DATA EXPORTED AS AN XML FILE


import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import serial
import time
print ()
print ()


#Set COM port depending on user setup
serPort = 'COM6'
baudRate = 9600
ser = serial.Serial(serPort, baudRate)
startMarker = 60
endMarker = 62


def sendToArduino(sendStr):
    ser.write(sendStr.encode('utf-8')) # change for Python3


#======================================

def recvFromArduino():
    global startMarker, endMarker
    
    ck = ""
    x = "z" # any value that is not an end- or startMarker
    byteCount = -1 # to allow for the fact that the last increment will be one too many
    
    # wait for the start character
    while  ord(x) != startMarker: 
        x = ser.read()
    
    # save data until the end marker is found
    while ord(x) != endMarker:
        if ord(x) != startMarker:
            ck = ck + x.decode("utf-8") # change for Python3
            byteCount += 1
        x = ser.read()
    
    return(ck)


#============================

def waitForArduino():

    # wait until the Arduino sends 'Arduino Ready' - allows time for Arduino reset
    # it also ensures that any bytes left over from a previous message are discarded
    
    global startMarker, endMarker
    
    msg = ""
    while msg.find("Arduino is ready") == -1:

        while ser.inWaiting() == 0:
            pass
        
        msg = recvFromArduino()

        print (msg) # python3 requires parenthesis
        print ()
        
#======================================

def runTest(td):
    numLoops = len(td)
    waitingForReply = False

    n = 0
    while n < numLoops:
        teststr = td[n]

        if waitingForReply == False:
            sendToArduino(teststr)
            print ("Sent from PC -- LOOP NUM " + str(n) + " TEST STR " + teststr)
            waitingForReply = True

        if waitingForReply == True:

            while ser.inWaiting() == 0:
                pass
            
            dataRecvd = recvFromArduino()
            print ("Reply Received  " + dataRecvd)
            n += 1
            waitingForReply = False

            print ("===========")

        time.sleep(5)


#======================================


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))



# Funciton aimed at cleaning complicated exports from UV-Vis. XML provides the best
# format for data exports. Function will loop through the XML document and pull relevant
# information and re-export it to a cleaner excel format.
def clean_data(file_path):

    print('Importing Data_file...')
    tree = ET.parse(file_path)
    root = tree.getroot()

    print('---------------')



    platesection_df = pd.DataFrame(columns = ['Name','InstrumentInfo', 'ReadTime'])
    wells_df = pd.DataFrame(columns = ['WellID','Name','Row','Col'])
    wavelengths = []
    absorbances = []
    timedata = []
    # Moving to each PlateSection output. Here we can collect plate name and read time info
    # Additionally could add support for read mode and type 


    print('Cleaning data...')
    for section in root.findall('./PlateSections/PlateSection'):

        for node in section.iter('WavelengthSettings'):
            wave_temp = node.find('Wavelength').text


        for node in section.iter('Well'):
            #print(node.attrib)
            #print(node.find('RawData').text)
            el_check = ET.iselement(node.find('RawData'))
            if el_check == True:

                #Populating all of the data to corresponding dataframes at the index of each well
                platesection_df = platesection_df.append(section.attrib,ignore_index=True)
                wells_df = wells_df.append(node.attrib,ignore_index=True)



                absorbances.append(node.find('RawData').text)
                wavelengths.append(wave_temp)
                kinetic_check = ET.iselement('TimeData')
                if kinetic_check == True:
                    absorbances.append(node.find('TimeData').text)



    wells_df['RawData'] = absorbances
    wells_df['Wavelength'] = wavelengths
    if kinetic_check == True:
        wells_df['TimeData'] = timedata


    Data = pd.concat([wells_df, platesection_df], axis=1)
    Data.columns = ['WellID','Well Position','Row','Col','RawData','Wavelength','Plate','InstrumentInfo','ReadTime']
    Data = Data.drop(['Row','Col'],axis=1)

    excel_path = file_path.replace('.xml','.xlsx')



    Data.to_excel(excel_path)
    


    print('Cleaned File!')
    ##Remove background noise and create ratios only for wells that contain polymer
    i=0
    j=0
    k=0
    a615 = []
    a632 = []
    absorbances = list(map(float, absorbances))
    a615_raw = absorbances[0:96]
    for x in a615_raw:
        if x > 15:
            a615.append(a615_raw[i])
        else:
            a615.append(0)
        i = i+1

    a632_raw = absorbances[96:192]
    for x in a632_raw:
        if x > 15:
            a632.append(a632_raw[j])
        else:
            a632.append(0)
        j = j+1

    #print(a632)
    #print(a615)
    #Create array of ratios for wells
    ratio = []

    for x in a632:
        if x > 0:
            ratio.append(a632[k]/a615[k])
        else:
            ratio.append(0)
        k = k+1

    print(ratio)

    #Array of zeros needs to be initialized before doing data analysis
    #Run 'arraysave.py' to initiailize
    n = 0
    first = np.load('ratio.npy')
    print(first)
    slope_ratio = []
    for x in ratio:
        if x > 0:
            slope_ratio.append(((ratio[n]-first[n])/30))
        else:
            slope_ratio.append(0)
        n = n+1 

    np.save('ratio.npy', ratio)

    print(slope_ratio)


    #Create binary data to send to Arduino out of calculated ratio slopes
    thresh = []
    for x in slope_ratio:
     if ((x < .002) and (x > 0)) or (x<0):
         thresh.append(0)
     elif x ==0:
        thresh.append(0)
     else:
        thresh.append(1)
    
    
    #print(thresh)
    bits = [[1,2,4,8,16,32,64,128,256,512,1024,2048],
            [1,2,4,8,16,32,64,128,256,512,1024,2048],
            [1,2,4,8,16,32,64,128,256,512,1024,2048],
            [1,2,4,8,16,32,64,128,256,512,1024,2048],
            [1,2,4,8,16,32,64,128,256,512,1024,2048],
            [1,2,4,8,16,32,64,128,256,512,1024,2048],
            [1,2,4,8,16,32,64,128,256,512,1024,2048],
            [1,2,4,8,16,32,64,128,256,512,1024,2048]]
    #binary = thresh[96:192]

    #Create array that communicates with Arduino that tells it which LED's to turn on
    binary = np.array(thresh).reshape(8,12)
    byte_data = binary*bits
    print(byte_data)

    input_data = []
    rows = len(byte_data)
    cols = len(byte_data[0])
    total = 0
    for x in range(0,rows):
        rowtotal=0
        for y in range(0,cols):
            rowtotal=rowtotal+int(byte_data[x][y])
        input_data.append(rowtotal)



    print ("Serial port " + serPort + " opened  Baudrate " + str(baudRate))


    


    waitForArduino()
    
    b = '<'+ ','.join(map(str, input_data)) + '>'
    testData = [b]

    runTest(testData)


    ser.close()
 


clean_data(r'**FluorescenceXMLPath')

 
