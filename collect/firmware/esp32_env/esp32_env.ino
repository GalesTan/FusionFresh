/*
 * 板卡：esp32_env（蓝牙名 ESP32-Env）
 * 传感器：C2H5OH + C2H4 + DHT1 + DHT2
 *
 * C2H5OH  乙醇 Modbus UART2 RX=16 TX=17
 * C2H4    乙烯 模拟量 ADC GPIO34，0~20ppm  （已改为模拟量）
 * DHT1    DHT11 温湿度 GPIO4
 * DHT2    DHT11 温湿度 GPIO5
 *
 * 统一行协议（与 esp32_gas / Python bt_reader 共用）：
 *   SENSOR:value unit[|KEY:VAL]*
 * 多传感器空格分隔，以 \r\n 结尾。示例：
 *   C2H5OH:1.23 %LEL|STATUS:1 C2H4:0.45 ppm|V:0.123 DHT1_T:25.3 C DHT1_H:60 %RH DHT2_T:25.1 C DHT2_H:58 %RH
 *
 * 基于 4MZ-HH 模组规格书 V1.36（乙醇仍用Modbus）
 * 乙烯改用模拟量 0~2V -> 0~20ppm
 */

#include <ModbusMaster.h>
#include <BluetoothSerial.h>
#include <DHT.h>

BluetoothSerial SerialBT;

#define BT_NAME "ESP32-Env"

unsigned long lastPrintTime = 0;
const unsigned long PRINT_INTERVAL = 30000;   // 30 秒自动打印

// ==================== DHT11 配置 ====================
#define DHT_DATA_PIN_1 4
#define DHT_DATA_PIN_2 5
#define DHT_MODEL DHT11

DHT dht1(DHT_DATA_PIN_1, DHT_MODEL);
DHT dht2(DHT_DATA_PIN_2, DHT_MODEL);

float dht1_temperature = 0.0;
float dht1_humidity = 0.0;
float dht2_temperature = 0.0;
float dht2_humidity = 0.0;
bool dht1_ok = false;
bool dht2_ok = false;

unsigned long lastDHTRead = 0;
const unsigned long DHT_INTERVAL = 2000;

// ==================== 乙醇 C2H5OH UART2（Modbus）====================
#define ETH_RX_PIN 16
#define ETH_TX_PIN 17
#define ETH_SENSOR_ADDR 0x01
const uint16_t ETH_START_REG = 0x0000;
const uint16_t ETH_NUM_REGS = 0x000A;
ModbusMaster ethanolNode;

// ==================== 乙烯 C2H4 模拟量 ADC（GPIO34）====================
#define ETHYLENE_ADC_PIN    34          // 使用 GPIO34
#define ETHYLENE_MAX_VOLT   2.0f        // 模组最大输出电压 2V
#define ETHYLENE_FULL_RANGE 20.0f       // 量程 0~20ppm

// 滑动平均滤波（保留可选，窗口改为5以加快响应）
#define FILTER_WINDOW 5
float filterEthylene[FILTER_WINDOW];
uint8_t filterIndex = 0;

// ==================== 函数声明 ====================
bool appendEthanol(String &out);
void appendEthylene(String &out);
void appendDHT(String &out);
void sendUnifiedFrame();

// ==================== 读取乙醇（Modbus）====================
bool appendEthanol(String &out) {
  uint8_t result = ethanolNode.readHoldingRegisters(ETH_START_REG, ETH_NUM_REGS);
  if (result != ethanolNode.ku8MBSuccess) {
    return false;
  }

  uint16_t reg0 = ethanolNode.getResponseBuffer(0x00);
  uint8_t decimalPlaces = (reg0 >> 8) & 0x0F;
  uint16_t rawConcentration = ethanolNode.getResponseBuffer(0x01);
  uint16_t lowAlarm = ethanolNode.getResponseBuffer(0x02);
  uint16_t highAlarm = ethanolNode.getResponseBuffer(0x03);
  uint16_t fullScale = ethanolNode.getResponseBuffer(0x04);
  uint8_t status = ethanolNode.getResponseBuffer(0x05) & 0x0F;

  float concentration = rawConcentration;
  for (int i = 0; i < decimalPlaces; i++) concentration /= 10.0;

  out += "C2H5OH:" + String(concentration, 2) + " %LEL";
  out += "|STATUS:" + String(status);
  out += "|LOW:" + String(lowAlarm);
  out += "|HIGH:" + String(highAlarm);
  out += "|FS:" + String(fullScale);
  return true;
}

// ==================== 读取乙烯（模拟量）====================
void appendEthylene(String &out) {
  // 1. 读取 ADC
  uint16_t adc = analogRead(ETHYLENE_ADC_PIN);
  
  // 2. 计算电压 (ESP32 ADC 参考电压 3.3V, 12位)
  float voltage = adc * 3.3f / 4095.0f;
  
  // 3. 限幅（确保不超过 2V）
  if (voltage > ETHYLENE_MAX_VOLT) voltage = ETHYLENE_MAX_VOLT;
  if (voltage < 0) voltage = 0;
  
  // 4. 线性映射到浓度（0~2V → 0~20ppm）
  float concentration = (voltage / ETHYLENE_MAX_VOLT) * ETHYLENE_FULL_RANGE;
  
  // 5. 滑动平均滤波（窗口5）
  filterEthylene[filterIndex] = concentration;
  filterIndex = (filterIndex + 1) % FILTER_WINDOW;
  
  float avg = 0;
  for (int i = 0; i < FILTER_WINDOW; i++) {
    avg += filterEthylene[i];
  }
  avg /= FILTER_WINDOW;
  
  // 6. 拼接到输出字符串，同时附上原始电压
  out += "C2H4:" + String(avg, 3) + " ppm";   // 保留3位小数
  out += "|V:" + String(voltage, 3);
}

// ==================== 读取 DHT11 ====================
void appendDHT(String &out) {
  if (dht1_ok) {
    if (out.length() > 0) out += " ";
    out += "DHT1_T:" + String(dht1_temperature, 1) + " C";
    out += " DHT1_H:" + String(dht1_humidity, 0) + " %RH";
  }
  if (dht2_ok) {
    if (out.length() > 0) out += " ";
    out += "DHT2_T:" + String(dht2_temperature, 1) + " C";
    out += " DHT2_H:" + String(dht2_humidity, 0) + " %RH";
  }
}

// ==================== 统一发送帧 ====================
void sendUnifiedFrame() {
  String line;
  String part;

  // 1. 乙醇（Modbus）
  part = "";
  if (appendEthanol(part)) {
    line = part;
  }

  // 2. 乙烯（模拟量）
  part = "";
  appendEthylene(part);   // 总是尝试读取，不依赖返回值
  if (line.length() > 0) line += " ";
  line += part;

  // 3. DHT
  appendDHT(line);

  if (line.length() == 0) {
    SerialBT.println("# WARN no sensor data");
    return;
  }
  line += "\r\n";
  SerialBT.print(line);
}

// ==================== setup ====================
void setup() {
  // ---- UART0 留给乙醇？注意：原本乙烯占用 UART0，现在乙烯改为模拟，UART0 可释放，但为了兼容保留初始化，不再使用 ----
  // 因为乙烯已改为模拟，不再需要 UART0，但 Serial 仍可用于调试（如果需要），
  // 但按原设计 UART0 不打印调试，保持原样即可。
  // 我们不再初始化 ethyleneNode，因为不再使用。
  // 但为了保留原框架，我们仍然保留 Serial.begin，但不再绑定 ModbusMaster 对象。
  // 实际上我们可以完全删除 ethyleneNode 相关，但为了最小改动，将原来 ethyleneNode 的 begin 注释掉。
  Serial.begin(9600, SERIAL_8N1);
  // ethyleneNode.begin(ETHYLENE_SENSOR_ADDR, Serial);   // 已废弃，改用 ADC

  // 乙醇 UART2
  Serial2.begin(9600, SERIAL_8N1, ETH_RX_PIN, ETH_TX_PIN);
  ethanolNode.begin(ETH_SENSOR_ADDR, Serial2);

  // DHT
  dht1.begin();
  dht2.begin();

  // ADC 配置
  analogReadResolution(12);           // 12位分辨率
  analogSetAttenuation(ADC_11db);     // 量程 0~3.3V

  // 初始化滤波数组
  for (int i = 0; i < FILTER_WINDOW; i++) {
    filterEthylene[i] = 0.0;
  }

  // 蓝牙
  SerialBT.begin(BT_NAME);
  delay(2000);
  SerialBT.println("# esp32_env ready: C2H5OH (Modbus) + C2H4 (Analog) + DHT1 + DHT2");
  SerialBT.println("# protocol: SENSOR:value unit[|KEY:VAL]*");
  SerialBT.println("# cmds: read/start/stop/status/restart");
}

// ==================== loop ====================
void loop() {
  static bool autoPrintEnable = true;

  // DHT 定时读取
  if (millis() - lastDHTRead >= DHT_INTERVAL) {
    lastDHTRead = millis();

    float t1 = dht1.readTemperature();
    float h1 = dht1.readHumidity();
    if (!isnan(t1) && !isnan(h1)) {
      dht1_temperature = t1;
      dht1_humidity = h1;
      dht1_ok = true;
    }

    float t2 = dht2.readTemperature();
    float h2 = dht2.readHumidity();
    if (!isnan(t2) && !isnan(h2)) {
      dht2_temperature = t2;
      dht2_humidity = h2;
      dht2_ok = true;
    }
  }

  // 蓝牙命令处理
  if (SerialBT.available() > 0) {
    String cmd = SerialBT.readStringUntil('\n');
    cmd.trim();

    if (cmd == "read") {
      sendUnifiedFrame();
    } else if (cmd == "stop") {
      autoPrintEnable = false;
      SerialBT.println("# auto print OFF");
    } else if (cmd == "start") {
      autoPrintEnable = true;
      SerialBT.println("# auto print ON 30s");
    } else if (cmd == "status") {
      SerialBT.println("# board=esp32_env name=ESP32-Env");
      SerialBT.println("# C2H5OH addr=0x01 UART2 (Modbus)");
      SerialBT.println("# C2H4 ADC GPIO34 0-2V -> 0-20ppm");
      SerialBT.println("# DHT1 GPIO4 | DHT2 GPIO5");
    } else if (cmd == "restart") {
      SerialBT.println("# restarting...");
      delay(500);
      ESP.restart();
    }
  }

  // 自动打印
  if (autoPrintEnable && (millis() - lastPrintTime >= PRINT_INTERVAL)) {
    lastPrintTime = millis();
    sendUnifiedFrame();
  }
}