const int NUM_ROWS = 8;
const int NUM_COLS = 1;

const int dataPin = 8;
const int clockPin = 9;
const int latchPin=10;
int input_data[NUM_COLS][NUM_ROWS];
bool low_high[3];
byte bytedata[48];


int even_dataset[4][3] {
  {1,1,1},{0,0,0},{2,2,2},{0,0,0}
};

const byte numLEDs = 2;
byte ledPin[numLEDs] = {12, 13};
unsigned long LEDinterval[numLEDs] = {200, 400};
unsigned long prevLEDmillis[numLEDs] = {0, 0};

const byte buffSize = 40;
char inputBuffer[buffSize];
const char startMarker = '<';
const char endMarker = '>';
byte bytesRecvd = 0;
boolean readInProgress = false;
boolean newDataFromPC = false;

char messageFromPC[buffSize] = {0};
int newFlashInterval[12] {0,0,0,0,0,0,0,0,0,0,0,0};
float servoFraction = 0.0; // fraction of servo range to move

int data1_1 = 0;
int data1_2 = 0;
int data1_3 = 0;
int data2_1 = 0;
int data2_2 = 0;
int data2_3 = 0;
int data3_1 = 0;
int data3_2 = 0;
int data3_3 = 0;
int data4_1 = 0;
int data4_2 = 0;
int data4_3 = 0;

//int input_data[4][3] {{0,0,0},{0,0,0},{0,0,0},{0,0,0}};


unsigned long curMillis;

unsigned long prevReplyToPCmillis = 0;
unsigned long replyToPCinterval = 1000;

//=============

void setup() {
  Serial.begin(9600);
  
  delay(500);
  

  Serial.println("<Arduino is ready>");
  pinMode(clockPin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  pinMode(latchPin, OUTPUT);
 
}


//=============

void loop() {
  curMillis = millis();
  getDataFromPC();
  replyToPC();
  int i=0;
  displayPixels(i,100);
  i++;
  if (i==NUM_ROWS) {i=1;}
  
}

//=============

void getDataFromPC() {
    
  if(Serial.available() > 0) {

    char x = Serial.read();
      
    if (x == endMarker) {
      readInProgress = false;
      newDataFromPC = true;
      //inputBuffer[bytesRecvd] = 0;
      parseData();
    }
    
    if(readInProgress) {
      inputBuffer[bytesRecvd] = x;
      bytesRecvd ++;
      if (bytesRecvd == buffSize) {
        bytesRecvd = buffSize - 1;
      }
    }

    if (x == startMarker) { 
      bytesRecvd = 0; 
      readInProgress = true;
    }
  }
}

//=============

void getDataFromPC2() {
    
  char input_data[15];
  if(Serial.available() >0) {
    for (int i=0; i<15; i++) {
      input_data[i] = Serial.read();
    }
    Serial.println(input_data);
  }
delay(100);
}

//=============
 
void parseData(){

   char * strtokIndx; // this is used by strtok() as an index
   char *strings[NUM_COLS][NUM_ROWS];
   char *ptr = NULL;

  
   ptr = strtok(inputBuffer,",");      // get the first part - the string
   while(ptr != NULL) {
    for(int i = 0; i < NUM_COLS; i++) {
     for (int j = 0; j<NUM_ROWS; j++) {
      strings[i][j] = ptr;  
      ptr = strtok(NULL, ",");
      //Serial.println(strings[i][j]); 
   }
  for(int i = 0; i < NUM_COLS; i++) {
     for (int j = 0; j<NUM_ROWS; j++) { 
    input_data[i][j] = atoi(strings[i][j]);
     }}

      


}
   }
}



//=============

void replyToPC(){

  if (newDataFromPC) {
    newDataFromPC = true;
    Serial.print("<Msg ");
    for(int i = 0; i < NUM_COLS; i++) {
     for (int j = 0; j<NUM_ROWS; j++) {
       Serial.println(input_data[i][j]);
    }
    }
    
    //Serial.print(curMillis >> 9); // divide by 512 is approx = half-seconds
    Serial.print(">");
  }}
  
void displayPixels(int val, int delay) {
  for (int i=0; i<delay; i++)
  {
    for(int row=0; row<NUM_ROWS; row++){
      int gcur = (input_data[val][row] & 65280)/256;
      int rcur = (input_data[val][row] & 255);
      digitalWrite(latchPin,LOW);
      shiftOut(dataPin,clockPin, MSBFIRST, 255-gcur);
      shiftOut(dataPin,clockPin, MSBFIRST, 255-rcur);
      shiftOut(dataPin,clockPin, MSBFIRST, B00000001 <<row);
      digitalWrite(latchPin, HIGH);
      
      delayMicroseconds(50);
      digitalWrite(latchPin,LOW);
      shiftOut(dataPin,clockPin, LSBFIRST, 255);
      shiftOut(dataPin,clockPin, LSBFIRST, 255);
      shiftOut(dataPin,clockPin, LSBFIRST, B00000001 <<row);
      digitalWrite(latchPin,HIGH);
      
    }
  }
}