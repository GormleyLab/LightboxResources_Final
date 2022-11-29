#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import serial
import time


# In[2]:


def A_button_press(num):
    global A_equation_text
    A_equation_text = A_equation_text + str(num)
    A_equation_label.set(A_equation_text)

def A_equals():
    global A_equation_text
    A_equation_text = A_equation_text[:-3]
    A_total = str(eval(A_equation_text))
    A_equation_label.set(A_total)
    A_equation_text=A_total
def A_clear():
    global A_equation_text
    A_equation_label.set("0")
    A_equation_text = ""
    
def B_button_press(num):
    global B_equation_text
    B_equation_text = B_equation_text + str(num)
    B_equation_label.set(B_equation_text)

def B_equals():
    global B_equation_text
    B_equation_text = B_equation_text[:-3]
    B_total = str(eval(B_equation_text))
    B_equation_label.set(B_total)
    B_equation_text=B_total
def B_clear():
    global B_equation_text
    B_equation_label.set("0")
    B_equation_text = ""
    
def C_button_press(num):
    global C_equation_text
    C_equation_text = C_equation_text + str(num)
    C_equation_label.set(C_equation_text)

def C_equals():
    global C_equation_text
    C_equation_text = C_equation_text[:-3]
    C_total = str(eval(C_equation_text))
    C_equation_label.set(C_total)
    C_equation_text=C_total
def C_clear():
    global C_equation_text
    C_equation_label.set("0")
    C_equation_text = ""
    
def D_button_press(num):
    global D_equation_text
    D_equation_text = D_equation_text + str(num)
    D_equation_label.set(D_equation_text)

def D_equals():
    global D_equation_text
    D_equation_text = D_equation_text[:-3]
    D_total = str(eval(D_equation_text))
    D_equation_label.set(D_total)
    D_equation_text=D_total
def D_clear():
    global D_equation_text
    D_equation_label.set("0")
    D_equation_text = ""
    
def E_button_press(num):
    global E_equation_text
    E_equation_text = E_equation_text + str(num)
    E_equation_label.set(E_equation_text)

def E_equals():
    global E_equation_text
    E_equation_text = E_equation_text[:-3]
    E_total = str(eval(E_equation_text))
    E_equation_label.set(E_total)
    E_equation_text=E_total
def E_clear():
    global E_equation_text
    E_equation_label.set("0")
    E_equation_text = ""
    
def F_button_press(num):
    global F_equation_text
    F_equation_text = F_equation_text + str(num)
    F_equation_label.set(F_equation_text)

def F_equals():
    global F_equation_text
    F_equation_text = F_equation_text[:-3]
    F_total = str(eval(F_equation_text))
    F_equation_label.set(F_total)
    F_equation_text=F_total
def F_clear():
    global F_equation_text
    F_equation_label.set("0")
    F_equation_text = ""
    
def G_button_press(num):
    global G_equation_text
    G_equation_text = G_equation_text + str(num)
    G_equation_label.set(G_equation_text)

def G_equals():
    global G_equation_text
    G_equation_text = G_equation_text[:-3]
    G_total = str(eval(G_equation_text))
    G_equation_label.set(G_total)
    G_equation_text=G_total
def G_clear():
    global G_equation_text
    G_equation_label.set("0")
    G_equation_text = ""
    
def H_button_press(num):
    global H_equation_text
    H_equation_text = H_equation_text + str(num)
    H_equation_label.set(H_equation_text)

def H_equals():
    global H_equation_text
    H_equation_text = H_equation_text[:-3]
    H_total = str(eval(H_equation_text))
    H_equation_label.set(H_total)
    H_equation_text=H_total
def H_clear():
    global H_equation_text
    H_equation_label.set("0")
    H_equation_text = ""

def test():
    global A_equation_label,B_equation_label,C_equation_label,D_equation_label,E_equation_label,F_equation_label,G_equation_label,H_equation_label
    Data = [int(A_equation_label.get()),int(B_equation_label.get()),int(C_equation_label.get()),int(D_equation_label.get()),
            int(E_equation_label.get()),int(F_equation_label.get()),int(G_equation_label.get()),int(H_equation_label.get())]
    print(Data)
    waitForArduino()
    a = '<'+ ','.join(map(str, Data)) + '>'
    testData = [a]
    
    runTest(testData)
    
    ser.close()
    


# In[3]:


serPort = 'COM4'
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


# In[4]:


window = Tk()
window.geometry("1900x2000")
window.title("Lightbox Calculator")
window.configure(bg = "grey")

A_equation_text = ""
B_equation_text = ""
C_equation_text = ""
D_equation_text = ""
E_equation_text = ""
F_equation_text = ""
G_equation_text = ""
H_equation_text = ""

A_equation_label = StringVar()
A_equation_label.set("0")
B_equation_label = StringVar()
B_equation_label.set("0")
C_equation_label = StringVar()
C_equation_label.set("0")
D_equation_label = StringVar()
D_equation_label.set("0")
E_equation_label = StringVar()
E_equation_label.set("0")
F_equation_label = StringVar()
F_equation_label.set("0")
G_equation_label = StringVar()
G_equation_label.set("0")
H_equation_label = StringVar()
H_equation_label.set("0")

frame0 = Frame(window)
frame = Frame(window)
frame2 = Frame(window)
frame3 = Frame(window)
frame4 = Frame(window)
frame5 = Frame(window)
frame6 = Frame(window)
frame7 = Frame(window)
frame8 = Frame(window)
frame9 = Frame(window)
frame10 = Frame(window)
frame11 = Frame(window)
frame12 = Frame(window)
frame13 = Frame(window)
frame14 = Frame(window)
frame15 = Frame(window)
frame16 = Frame(window)

frame0.pack()
frame.pack()
frame2.pack()
frame3.pack()
frame4.pack()
frame5.pack()
frame6.pack()
frame7.pack()
frame8.pack()
frame9.pack()
frame10.pack()
frame11.pack()
frame12.pack()
frame13.pack()
frame14.pack()
frame15.pack()
frame16.pack()

Alabel = Label(frame0, textvariable = A_equation_label, font = ('consolas',10), bg = "white", width = 150, height = 2)
Alabel.pack(side=TOP)
Blabel = Label(frame2, textvariable = B_equation_label, font = ('consolas',10), bg = "white", width = 150, height = 2)
Blabel.grid(row=0,column=0)
Clabel = Label(frame4, textvariable = C_equation_label, font = ('consolas',10), bg = "white", width = 150, height = 2)
Clabel.grid(row=0,column=0)
Dlabel = Label(frame6, textvariable = D_equation_label, font = ('consolas',10), bg = "white", width = 150, height = 2)
Dlabel.grid(row=0,column=0)
Elabel = Label(frame8, textvariable = E_equation_label, font = ('consolas',10), bg = "white", width = 150, height = 2)
Elabel.grid(row=0,column=0)
Flabel = Label(frame10, textvariable = F_equation_label, font = ('consolas',10), bg = "white", width = 150, height = 2)
Flabel.grid(row=0,column=0)
Glabel = Label(frame12, textvariable = G_equation_label, font = ('consolas',10), bg = "white", width = 150, height = 2)
Glabel.grid(row=0,column=0)
Hlabel = Label(frame14, textvariable = H_equation_label, font = ('consolas',10), bg = "white", width = 150, height = 2)
Hlabel.grid(row=0,column=0)



A_button1 = Button(frame, text = 'A1', height = 2, width =9, font =35,
                command = lambda: A_button_press('1 + '), activebackground="green")
A_button1.grid(row=0, column =0)

A_button2 = Button(frame, text = 'A2', height = 2, width =9, font =35,
                command = lambda: A_button_press('2 + '), activebackground="green")
A_button2.grid(row=0, column =1)

A_button3 = Button(frame, text = 'A3', height = 2, width =9, font =35,
                command = lambda: A_button_press('4 + '), activebackground="green")
A_button3.grid(row=0, column =2)

A_button4 = Button(frame, text = 'A4', height = 2, width =9, font =35,
                command = lambda: A_button_press('8 + '), activebackground="green")
A_button4.grid(row=0, column =3)

A_button5 = Button(frame, text = 'A5', height = 2, width =9, font =35,
                command = lambda: A_button_press('16 + '), activebackground="green")
A_button5.grid(row=0, column =4)

A_button6 = Button(frame, text = 'A6', height = 2, width =9, font =35,
                command = lambda: A_button_press('32 + '), activebackground="green")
A_button6.grid(row=0, column =5)

A_button7 = Button(frame, text = 'A7', height = 2, width =9, font =35,
                command = lambda: A_button_press('64 + '), activebackground="green")
A_button7.grid(row=0, column =6)

A_button8 = Button(frame, text = 'A8', height = 2, width =9, font =35,
                command = lambda: A_button_press('128 + '), activebackground="green")
A_button8.grid(row=0, column =7)

A_button9 = Button(frame, text = 'A9', height = 2, width =9, font =35,
                command = lambda: A_button_press('256 + '), activebackground="green")
A_button9.grid(row=0, column =8)

A_button10 = Button(frame, text = 'A10', height = 2, width =9, font =35,
                command = lambda: A_button_press('512 + '), activebackground="green")
A_button10.grid(row=0, column =9)

A_button11 = Button(frame, text = 'A11', height = 2, width =9, font =35,
                command = lambda: A_button_press('1024 + '), activebackground="green")
A_button11.grid(row=0, column =10)

A_button12 = Button(frame, text = 'A12', height = 2, width =9, font =35,
                command = lambda: A_button_press('2048 + '), activebackground="green")
A_button12.grid(row=0, column =11)


A_equals_button = Button(frame, text = 'Total', height = 2, width =9, font =35, bg = "green",
                command = A_equals, activebackground="green")
A_equals_button.grid(row=0, column =12)

A_clear_button = Button(frame, text = 'Clear', height = 2, width =9, font =35, bg = "red",
                command = A_clear, activebackground="green")
A_clear_button.grid(row=0, column =13)



B_button1 = Button(frame3, text = 'B1', height = 2, width =9, font =35,
                command = lambda: B_button_press('1 + '), activebackground="green")
B_button1.grid(row=0, column =0)

B_button2 = Button(frame3, text = 'B2', height = 2, width =9, font =35,
                command = lambda: B_button_press('2 + '), activebackground="green")
B_button2.grid(row=0, column =1)

B_button3 = Button(frame3, text = 'B3', height = 2, width =9, font =35,
                command = lambda: B_button_press('4 + '), activebackground="green")
B_button3.grid(row=0, column =2)

B_button4 = Button(frame3, text = 'B4', height = 2, width =9, font =35,
                command = lambda: B_button_press('8 + '), activebackground="green")
B_button4.grid(row=0, column =3)

B_button5 = Button(frame3, text = 'B5', height = 2, width =9, font =35,
                command = lambda: B_button_press('16 + '), activebackground="green")
B_button5.grid(row=0, column =4)

B_button6 = Button(frame3, text = 'B6', height = 2, width =9, font =35,
                command = lambda: B_button_press('32 + '), activebackground="green")
B_button6.grid(row=0, column =5)

B_button7 = Button(frame3, text = 'B7', height = 2, width =9, font =35,
                command = lambda: B_button_press('64 + '), activebackground="green")
B_button7.grid(row=0, column =6)

B_button8 = Button(frame3, text = 'B8', height = 2, width =9, font =35,
                command = lambda: B_button_press('128 + '), activebackground="green")
B_button8.grid(row=0, column =7)

B_button9 = Button(frame3, text = 'B9', height = 2, width =9, font =35,
                command = lambda: B_button_press('256 + '), activebackground="green")
B_button9.grid(row=0, column =8)

B_button10 = Button(frame3, text = 'B10', height = 2, width =9, font =35,
                command = lambda: B_button_press('512 + '), activebackground="green")
B_button10.grid(row=0, column =9)

B_button11 = Button(frame3, text = 'B11', height = 2, width =9, font =35,
                command = lambda: B_button_press('1024 + '), activebackground="green")
B_button11.grid(row=0, column =10)

B_button12 = Button(frame3, text = 'B12', height = 2, width =9, font =35,
                command = lambda: B_button_press('2048 + '), activebackground="green")
B_button12.grid(row=0, column =11)


B_equals_button = Button(frame3, text = 'Total', height = 2, width =9, font =35, bg = "green",
                command = B_equals, activebackground="green")
B_equals_button.grid(row=0, column =12)

B_clear_button = Button(frame3, text = 'Clear', height = 2, width =9, font =35, bg = "red",
                command = B_clear, activebackground="green")
B_clear_button.grid(row=0, column =13)


C_button1 = Button(frame5, text = 'C1', height = 2, width =9, font =35,
                command = lambda: C_button_press('1 + '), activebackground="green")
C_button1.grid(row=0, column =0)

C_button2 = Button(frame5, text = 'C2', height = 2, width =9, font =35,
                command = lambda: C_button_press('2 + '), activebackground="green")
C_button2.grid(row=0, column =1)

C_button3 = Button(frame5, text = 'C3', height = 2, width =9, font =35,
                command = lambda: C_button_press('4 + '), activebackground="green")
C_button3.grid(row=0, column =2)

C_button4 = Button(frame5, text = 'C4', height = 2, width =9, font =35,
                command = lambda: C_button_press('8 + '), activebackground="green")
C_button4.grid(row=0, column =3)

C_button5 = Button(frame5, text = 'C5', height = 2, width =9, font =35,
                command = lambda: C_button_press('16 + '), activebackground="green")
C_button5.grid(row=0, column =4)

C_button6 = Button(frame5, text = 'C6', height = 2, width =9, font =35,
                command = lambda: C_button_press('32 + '), activebackground="green")
C_button6.grid(row=0, column =5)

C_button7 = Button(frame5, text = 'C7', height = 2, width =9, font =35,
                command = lambda: C_button_press('64 + '), activebackground="green")
C_button7.grid(row=0, column =6)

C_button8 = Button(frame5, text = 'C8', height = 2, width =9, font =35,
                command = lambda: C_button_press('128 + '), activebackground="green")
C_button8.grid(row=0, column =7)

C_button9 = Button(frame5, text = 'C9', height = 2, width =9, font =35,
                command = lambda: C_button_press('256 + '), activebackground="green")
C_button9.grid(row=0, column =8)

C_button10 = Button(frame5, text = 'C10', height = 2, width =9, font =35,
                command = lambda: C_button_press('512 + '), activebackground="green")
C_button10.grid(row=0, column =9)

C_button11 = Button(frame5, text = 'C11', height = 2, width =9, font =35,
                command = lambda: C_button_press('1024 + '), activebackground="green")
C_button11.grid(row=0, column =10)

C_button12 = Button(frame5, text = 'C12', height = 2, width =9, font =35,
                command = lambda: C_button_press('2048 + '), activebackground="green")
C_button12.grid(row=0, column =11)


C_equals_button = Button(frame5, text = 'Total', height = 2, width =9, font =35, bg = "green",
                command = C_equals, activebackground="green")
C_equals_button.grid(row=0, column =12)

C_clear_button = Button(frame5, text = 'Clear', height = 2, width =9, font =35, bg = "red",
                command = C_clear, activebackground="green")
C_clear_button.grid(row=0, column =13)


D_button1 = Button(frame7, text = 'D1', height = 2, width =9, font =35,
                command = lambda: D_button_press('1 + '), activebackground="green")
D_button1.grid(row=0, column =0)

D_button2 = Button(frame7, text = 'D2', height = 2, width =9, font =35,
                command = lambda: D_button_press('2 + '), activebackground="green")
D_button2.grid(row=0, column =1)

D_button3 = Button(frame7, text = 'D3', height = 2, width =9, font =35,
                command = lambda: D_button_press('4 + '), activebackground="green")
D_button3.grid(row=0, column =2)

D_button4 = Button(frame7, text = 'D4', height = 2, width =9, font =35,
                command = lambda: D_button_press('8 + '), activebackground="green")
D_button4.grid(row=0, column =3)

D_button5 = Button(frame7, text = 'D5', height = 2, width =9, font =35,
                command = lambda: D_button_press('16 + '), activebackground="green")
D_button5.grid(row=0, column =4)

D_button6 = Button(frame7, text = 'D6', height = 2, width =9, font =35,
                command = lambda: D_button_press('32 + '), activebackground="green")
D_button6.grid(row=0, column =5)

D_button7 = Button(frame7, text = 'D7', height = 2, width =9, font =35,
                command = lambda: D_button_press('64 + '), activebackground="green")
D_button7.grid(row=0, column =6)

D_button8 = Button(frame7, text = 'D8', height = 2, width =9, font =35,
                command = lambda: D_button_press('128 + '), activebackground="green")
D_button8.grid(row=0, column =7)

D_button9 = Button(frame7, text = 'D9', height = 2, width =9, font =35,
                command = lambda: D_button_press('256 + '), activebackground="green")
D_button9.grid(row=0, column =8)

D_button10 = Button(frame7, text = 'D10', height = 2, width =9, font =35,
                command = lambda: D_button_press('512 + '), activebackground="green")
D_button10.grid(row=0, column =9)

D_button11 = Button(frame7, text = 'D11', height = 2, width =9, font =35,
                command = lambda: D_button_press('1024 + '), activebackground="green")
D_button11.grid(row=0, column =10)

D_button12 = Button(frame7, text = 'D12', height = 2, width =9, font =35,
                command = lambda: D_button_press('2048 + '), activebackground="green")
D_button12.grid(row=0, column =11)


D_equals_button = Button(frame7, text = 'Total', height = 2, width =9, font =35, bg = "green",
                command = D_equals, activebackground="green")
D_equals_button.grid(row=0, column =12)

D_clear_button = Button(frame7, text = 'Clear', height = 2, width =9, font =35, bg = "red",
                command = D_clear, activebackground="green")
D_clear_button.grid(row=0, column =13)


E_button1 = Button(frame9, text = 'E1', height = 2, width =9, font =35,
                command = lambda: E_button_press('1 + '), activebackground="green")
E_button1.grid(row=0, column =0)

E_button2 = Button(frame9, text = 'E2', height = 2, width =9, font =35,
                command = lambda: E_button_press('2 + '), activebackground="green")
E_button2.grid(row=0, column =1)

E_button3 = Button(frame9, text = 'E3', height = 2, width =9, font =35,
                command = lambda: E_button_press('4 + '), activebackground="green")
E_button3.grid(row=0, column =2)

E_button4 = Button(frame9, text = 'E4', height = 2, width =9, font =35,
                command = lambda: E_button_press('8 + '), activebackground="green")
E_button4.grid(row=0, column =3)

E_button5 = Button(frame9, text = 'E5', height = 2, width =9, font =35,
                command = lambda: E_button_press('16 + '), activebackground="green")
E_button5.grid(row=0, column =4)

E_button6 = Button(frame9, text = 'E6', height = 2, width =9, font =35,
                command = lambda: E_button_press('32 + '), activebackground="green")
E_button6.grid(row=0, column =5)

E_button7 = Button(frame9, text = 'E7', height = 2, width =9, font =35,
                command = lambda: E_button_press('64 + '), activebackground="green")
E_button7.grid(row=0, column =6)

E_button8 = Button(frame9, text = 'E8', height = 2, width =9, font =35,
                command = lambda: E_button_press('128 + '), activebackground="green")
E_button8.grid(row=0, column =7)

E_button9 = Button(frame9, text = 'E9', height = 2, width =9, font =35,
                command = lambda: E_button_press('256 + '), activebackground="green")
E_button9.grid(row=0, column =8)

E_button10 = Button(frame9, text = 'E10', height = 2, width =9, font =35,
                command = lambda: E_button_press('512 + '), activebackground="green")
E_button10.grid(row=0, column =9)

E_button11 = Button(frame9, text = 'E11', height = 2, width =9, font =35,
                command = lambda: E_button_press('1024 + '), activebackground="green")
E_button11.grid(row=0, column =10)

E_button12 = Button(frame9, text = 'E12', height = 2, width =9, font =35,
                command = lambda: E_button_press('2048 + '), activebackground="green")
E_button12.grid(row=0, column =11)


E_equals_button = Button(frame9, text = 'Total', height = 2, width =9, font =35, bg = "green",
                command = E_equals, activebackground="green")
E_equals_button.grid(row=0, column =12)

E_clear_button = Button(frame9, text = 'Clear', height = 4, width =9, font =35, bg = "red",
                command = E_clear, activebackground="green")
E_clear_button.grid(row=0, column =13)


F_button1 = Button(frame11, text = 'F1', height = 2, width =9, font =35,
                command = lambda: F_button_press('1 + '), activebackground="green")
F_button1.grid(row=0, column =0)

F_button2 = Button(frame11, text = 'F2', height = 2, width =9, font =35,
                command = lambda: F_button_press('2 + '), activebackground="green")
F_button2.grid(row=0, column =1)

F_button3 = Button(frame11, text = 'F3', height = 2, width =9, font =35,
                command = lambda: F_button_press('4 + '), activebackground="green")
F_button3.grid(row=0, column =2)

F_button4 = Button(frame11, text = 'F4', height = 2, width =9, font =35,
                command = lambda: F_button_press('8 + '), activebackground="green")
F_button4.grid(row=0, column =3)

F_button5 = Button(frame11, text = 'F5', height = 2, width =9, font =35,
                command = lambda: F_button_press('16 + '), activebackground="green")
F_button5.grid(row=0, column =4)

F_button6 = Button(frame11, text = 'F6', height = 2, width =9, font =35,
                command = lambda: F_button_press('32 + '), activebackground="green")
F_button6.grid(row=0, column =5)

F_button7 = Button(frame11, text = 'F7', height = 2, width =9, font =35,
                command = lambda: F_button_press('64 + '), activebackground="green")
F_button7.grid(row=0, column =6)

F_button8 = Button(frame11, text = 'F8', height = 2, width =9, font =35,
                command = lambda: F_button_press('128 + '), activebackground="green")
F_button8.grid(row=0, column =7)

F_button9 = Button(frame11, text = 'F9', height = 2, width =9, font =35,
                command = lambda: F_button_press('256 + '), activebackground="green")
F_button9.grid(row=0, column =8)

F_button10 = Button(frame11, text = 'F10', height = 2, width =9, font =35,
                command = lambda: F_button_press('512 + '), activebackground="green")
F_button10.grid(row=0, column =9)

F_button11 = Button(frame11, text = 'F11', height = 2, width =9, font =35,
                command = lambda: F_button_press('1024 + '), activebackground="green")
F_button11.grid(row=0, column =10)

F_button12 = Button(frame11, text = 'F12', height = 2, width =9, font =35,
                command = lambda: F_button_press('2048 + '), activebackground="green")
F_button12.grid(row=0, column =11)


F_equals_button = Button(frame11, text = 'Total', height = 2, width =9, font =35, bg = "green",
                command = F_equals, activebackground="green")
F_equals_button.grid(row=0, column =12)

F_clear_button = Button(frame11, text = 'Clear', height = 2, width =9, font =35, bg = "red",
                command = F_clear, activebackground="green")
F_clear_button.grid(row=0, column =13)


G_button1 = Button(frame13, text = 'G1', height = 2, width =9, font =35,
                command = lambda: G_button_press('1 + '), activebackground="green")
G_button1.grid(row=0, column =0)

G_button2 = Button(frame13, text = 'G2', height = 2, width =9, font =35,
                command = lambda: G_button_press('2 + '), activebackground="green")
G_button2.grid(row=0, column =1)

G_button3 = Button(frame13, text = 'G3', height = 2, width =9, font =35,
                command = lambda: G_button_press('4 + '), activebackground="green")
G_button3.grid(row=0, column =2)

G_button4 = Button(frame13, text = 'G4', height = 2, width =9, font =35,
                command = lambda: G_button_press('8 + '), activebackground="green")
G_button4.grid(row=0, column =3)

G_button5 = Button(frame13, text = 'G5', height = 2, width =9, font =35,
                command = lambda: G_button_press('16 + '), activebackground="green")
G_button5.grid(row=0, column =4)

G_button6 = Button(frame13, text = 'G6', height = 2, width =9, font =35,
                command = lambda: G_button_press('32 + '), activebackground="green")
G_button6.grid(row=0, column =5)

G_button7 = Button(frame13, text = 'G7', height = 2, width =9, font =35,
                command = lambda: G_button_press('64 + '), activebackground="green")
G_button7.grid(row=0, column =6)

G_button8 = Button(frame13, text = 'G8', height = 2, width =9, font =35,
                command = lambda: G_button_press('128 + '), activebackground="green")
G_button8.grid(row=0, column =7)

G_button9 = Button(frame13, text = 'G9', height = 2, width =9, font =35,
                command = lambda: G_button_press('256 + '), activebackground="green")
G_button9.grid(row=0, column =8)

G_button10 = Button(frame13, text = 'G10', height = 2, width =9, font =35,
                command = lambda: G_button_press('512 + '), activebackground="green")
G_button10.grid(row=0, column =9)

G_button11 = Button(frame13, text = 'G11', height = 2, width =9, font =35,
                command = lambda: G_button_press('1024 + '), activebackground="green")
G_button11.grid(row=0, column =10)

G_button12 = Button(frame13, text = 'G12', height = 2, width =9, font =35,
                command = lambda: G_button_press('2048 + '), activebackground="green")
G_button12.grid(row=0, column =11)


G_equals_button = Button(frame13, text = 'Total', height = 2, width =9, font =35, bg = "green",
                command = G_equals, activebackground="green")
G_equals_button.grid(row=0, column =12)

G_clear_button = Button(frame13, text = 'Clear', height = 2, width =9, font =35, bg = "red",
                command = G_clear, activebackground="green")
G_clear_button.grid(row=0, column =13)


H_button1 = Button(frame15, text = 'H1', height = 2, width =9, font =35,
                command = lambda: H_button_press('1 + '), activebackground="green")
H_button1.grid(row=0, column =0)

H_button2 = Button(frame15, text = 'H2', height = 2, width =9, font =35,
                command = lambda: H_button_press('2 + '), activebackground="green")
H_button2.grid(row=0, column =1)

H_button3 = Button(frame15, text = 'H3', height = 2, width =9, font =35,
                command = lambda: H_button_press('4 + '), activebackground="green")
H_button3.grid(row=0, column =2)

H_button4 = Button(frame15, text = 'H4', height = 2, width =9, font =35,
                command = lambda: H_button_press('8 + '), activebackground="green")
H_button4.grid(row=0, column =3)

H_button5 = Button(frame15, text = 'H5', height = 2, width =9, font =35,
                command = lambda: H_button_press('16 + '), activebackground="green")
H_button5.grid(row=0, column =4)

H_button6 = Button(frame15, text = 'H6', height = 2, width =9, font =35,
                command = lambda: H_button_press('32 + '), activebackground="green")
H_button6.grid(row=0, column =5)

H_button7 = Button(frame15, text = 'H7', height = 2, width =9, font =35,
                command = lambda: H_button_press('64 + '), activebackground="green")
H_button7.grid(row=0, column =6)

H_button8 = Button(frame15, text = 'H8', height = 2, width =9, font =35,
                command = lambda: H_button_press('128 + '), activebackground="green")
H_button8.grid(row=0, column =7)

H_button9 = Button(frame15, text = 'H9', height = 2, width =9, font =35,
                command = lambda: H_button_press('256 + '), activebackground="green")
H_button9.grid(row=0, column =8)

H_button10 = Button(frame15, text = 'H10', height = 2, width =9, font =35,
                command = lambda: H_button_press('512 + '), activebackground="green")
H_button10.grid(row=0, column =9)

H_button11 = Button(frame15, text = 'H11', height = 2, width =9, font =35,
                command = lambda: H_button_press('1024 + '), activebackground="green")
H_button11.grid(row=0, column =10)

H_button12 = Button(frame15, text = 'H12', height = 2, width =9, font =35,
                command = lambda: H_button_press('2048 + '), activebackground="green")
H_button12.grid(row=0, column =11)


H_equals_button = Button(frame15, text = 'Total', height = 2, width =9, font =35, bg = "green",
                command = H_equals, activebackground="green")
H_equals_button.grid(row=0, column =12)

H_clear_button = Button(frame15, text = 'Clear', height = 2, width =9, font =35, bg = "red",
                command = H_clear, activebackground="green")
H_clear_button.grid(row=0, column =13)

test_button = Button(frame16, text = 'Test Lightbox', height = 4, width = 20, font = 35, bg = "green",
                    command = test, activebackground="red")
test_button.grid(row = 0, column = 0)






window.mainloop()


# In[ ]:





# ## 
