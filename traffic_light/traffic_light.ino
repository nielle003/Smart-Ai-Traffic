/*
  4-Way AI Traffic Light Controller — with Pedestrian Buttons
  =============================================================
  Traffic lights unchanged: pins 2-13
    North: 2=RED  3=YELLOW  4=GREEN
    South: 5=RED  6=YELLOW  7=GREEN
    East:  8=RED  9=YELLOW  10=GREEN
    West:  11=RED 12=YELLOW 13=GREEN

  Pedestrian buttons: analog pins used as digital inputs
    A0 = North pedestrian button
    A1 = South pedestrian button
    A2 = East  pedestrian button
    A3 = West  pedestrian button

  Wiring each button (breadboard):
    One leg → Arduino analog pin (A0/A1/A2/A3)
    Other leg → GND rail → Arduino GND
    No resistor needed — INPUT_PULLUP is enabled in code

  Serial commands received (9600 baud):
    NS:RED  NS:YELLOW  NS:GREEN
    EW:RED  EW:YELLOW  EW:GREEN
    BOTH:RED  PEDESTRIAN

  Serial messages sent automatically when button pressed:
    BTN:N   BTN:S   BTN:E   BTN:W

  Responds "OK" after every command received.
*/

// ── Traffic light pins ────────────────────────────────────────────────────────
const int N_RED = 2,  N_YEL = 3,  N_GRN = 4;
const int S_RED = 5,  S_YEL = 6,  S_GRN = 7;
const int E_RED = 8,  E_YEL = 9,  E_GRN = 10;
const int W_RED = 11, W_YEL = 12, W_GRN = 13;

// ── Button pins ───────────────────────────────────────────────────────────────
const int BTN_N = A0;
const int BTN_S = A1;
const int BTN_E = A2;
const int BTN_W = A3;

// ── Debounce state ────────────────────────────────────────────────────────────
bool lastN = HIGH, lastS = HIGH, lastE = HIGH, lastW = HIGH;
unsigned long debounceN=0, debounceS=0, debounceE=0, debounceW=0;
const unsigned long DEBOUNCE_MS = 50;

String buf = "";

void setup() {
  Serial.begin(9600);

  // Traffic light pins as outputs
  int pins[] = {N_RED,N_YEL,N_GRN, S_RED,S_YEL,S_GRN,
                E_RED,E_YEL,E_GRN, W_RED,W_YEL,W_GRN};
  for (int i = 0; i < 12; i++) {
    pinMode(pins[i], OUTPUT);
    digitalWrite(pins[i], LOW);
  }

  // Button pins: INPUT_PULLUP means not pressed=HIGH, pressed=LOW
  pinMode(BTN_N, INPUT_PULLUP);
  pinMode(BTN_S, INPUT_PULLUP);
  pinMode(BTN_E, INPUT_PULLUP);
  pinMode(BTN_W, INPUT_PULLUP);

  setNS("RED");
  setEW("RED");
  Serial.println("READY");
}

void loop() {
  // Read serial commands from Python
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf.trim();
      if (buf.length() > 0) handleCommand(buf);
      buf = "";
    } else { buf += c; }
  }

  // Check pedestrian buttons
  checkBtn(BTN_N, lastN, debounceN, "BTN:N");
  checkBtn(BTN_S, lastS, debounceS, "BTN:S");
  checkBtn(BTN_E, lastE, debounceE, "BTN:E");
  checkBtn(BTN_W, lastW, debounceW, "BTN:W");
}

void checkBtn(int pin, bool &last, unsigned long &t, const char* msg) {
  bool cur = digitalRead(pin);
  if (cur == LOW && last == HIGH) {
    unsigned long now = millis();
    if (now - t > DEBOUNCE_MS) {
      Serial.println(msg);
      t = now;
    }
  }
  last = cur;
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
