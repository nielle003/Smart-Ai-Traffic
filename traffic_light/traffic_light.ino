/*
  4-Way AI Traffic Light Controller — 4 Cameras + 4 Lights
  ==========================================================
  One light per direction, each wired independently.
  N and S always receive the same command (NS axis).
  E and W always receive the same command (EW axis).

  Pin layout (12 pins, 2-13):
    North: pin 2=RED  pin 3=YELLOW  pin 4=GREEN
    South: pin 5=RED  pin 6=YELLOW  pin 7=GREEN
    East:  pin 8=RED  pin 9=YELLOW  pin 10=GREEN
    West:  pin 11=RED pin 12=YELLOW pin 13=GREEN

  Wiring each LED:
    Arduino pin → 220Ω resistor → LED anode (+)
    LED cathode (−) → GND

  Serial commands (9600 baud, newline terminated):
    NS:RED    NS:YELLOW    NS:GREEN
    EW:RED    EW:YELLOW    EW:GREEN
    BOTH:RED
    PEDESTRIAN

  Responds "OK" after every command.
*/

const int N_RED = 2,  N_YEL = 3,  N_GRN = 4;
const int S_RED = 5,  S_YEL = 6,  S_GRN = 7;
const int E_RED = 8,  E_YEL = 9,  E_GRN = 10;
const int W_RED = 11, W_YEL = 12, W_GRN = 13;

String buf = "";

void setup() {
  Serial.begin(9600);
  int pins[] = {N_RED,N_YEL,N_GRN, S_RED,S_YEL,S_GRN,
                E_RED,E_YEL,E_GRN, W_RED,W_YEL,W_GRN};
  for (int i = 0; i < 12; i++) {
    pinMode(pins[i], OUTPUT);
    digitalWrite(pins[i], LOW);
  }
  setNS("RED");
  setEW("RED");
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf.trim();
      if (buf.length() > 0) handleCommand(buf);
      buf = "";
    } else { buf += c; }
  }
}

void handleCommand(String cmd) {
  if      (cmd == "NS:RED")    { setNS("RED");    Serial.println("OK"); }
  else if (cmd == "NS:YELLOW") { setNS("YELLOW"); Serial.println("OK"); }
  else if (cmd == "NS:GREEN")  { setNS("GREEN");  Serial.println("OK"); }
  else if (cmd == "EW:RED")    { setEW("RED");    Serial.println("OK"); }
  else if (cmd == "EW:YELLOW") { setEW("YELLOW"); Serial.println("OK"); }
  else if (cmd == "EW:GREEN")  { setEW("GREEN");  Serial.println("OK"); }
  else if (cmd == "BOTH:RED")  { setNS("RED"); setEW("RED"); Serial.println("OK"); }
  else if (cmd == "PEDESTRIAN"){ setNS("RED"); setEW("RED"); Serial.println("OK"); }
  else { Serial.println("ERR:UNKNOWN"); }
}

// North + South always show same color
void setNS(String s) {
  setLight(N_RED, N_YEL, N_GRN, s);
  setLight(S_RED, S_YEL, S_GRN, s);
}

// East + West always show same color
void setEW(String s) {
  setLight(E_RED, E_YEL, E_GRN, s);
  setLight(W_RED, W_YEL, W_GRN, s);
}

void setLight(int r, int y, int g, String s) {
  digitalWrite(r, LOW); digitalWrite(y, LOW); digitalWrite(g, LOW);
  if      (s == "RED")    digitalWrite(r, HIGH);
  else if (s == "YELLOW") digitalWrite(y, HIGH);
  else if (s == "GREEN")  digitalWrite(g, HIGH);
}
