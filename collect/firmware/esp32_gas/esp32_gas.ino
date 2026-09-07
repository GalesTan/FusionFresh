/*
 * 板卡：esp32_gas（蓝牙名 ESP32-Gas）
 * 传感器：H2S + NH3 + VOC + CH3SH
 *
 * H2S     模拟量 ADC GPIO34，0~100ppm
 * NH3     模拟量 ADC GPIO35，0~200ppm
 * VOC     模拟量 ADC GPIO32，0~500ppm
 * CH3SH   Modbus UART0 RX0=GPIO3 TX0=GPIO1，0~100ppm
 *
 * 统一行协议（与 esp32_env / Python bt_reader 共用）：
 *   SENSOR:value unit[|KEY:VAL]*
 * 多传感器空格分隔，以 \r\n 结尾。示例：
 *   H2S:12.34 ppm|V:0.246 NH3:56.78 ppm|V:0.567 VOC:3.45 ppm|V:0.014 CH3SH:0.45 ppm|STATUS:1|LOW:20|HIGH:50|FS:100
 *
 * UART0 专供 CH3SH，勿在 Serial 上打印调试数据
 */

#include "BluetoothSerial.h"
#include <ModbusMaster.h>

BluetoothSerial SerialBT;

#define BT_NAME "ESP32-Gas"

// ==================== 硫化氢 H2S 模拟量 0~100ppm ====================
#define H2S_ADC_PIN     34
#define H2S_MAX_VOLT    2.0f
#define H2S_FULL_RANGE  100.0f

// ==================== 氨气 NH3 模拟量 0~200ppm ====================
#define NH3_ADC_PIN     35
#define NH3_MAX_VOLT    2.0f
#define NH3_FULL_RANGE  200.0f

// ==================== VOC 模拟量 0~500ppm（新增）====================
#define VOC_ADC_PIN     32          // 使用 GPIO32
#define VOC_MAX_VOLT    2.0f        // 模组最大输出电压 2V
#define VOC_FULL_RANGE  500.0f      // 量程 0~500ppm

// 滑动平均滤波（窗口5，平滑且响应较快）
#define FILTER_WINDOW   5
float filterVOC[FILTER_WINDOW];
uint8_t filterIndex = 0;

// ==================== CH3SH Modbus UART0 ====================
#define CH3SH_SENSOR_ADDR 0x01
const uint16_t CH3SH_START_REG = 0x0000;
const uint16_t CH3SH_NUM_REGS = 0x000A;
ModbusMaster ch3shNode;

#define READ_INTERVAL 30000UL   // 30 秒采样一次
unsigned long lastReadTime = 0;

// ==================== 公共函数：读取 Modbus 气体（仅用于 CH3SH）====================
bool appendModbusGas(ModbusMaster &node, uint16_t startReg, uint16_t numRegs,
                     const char *name, String &out) {
  uint8_t result = node.readHoldingRegisters(startReg, numRegs);
  if (result != node.ku8MBSuccess) {
    return false;
  }

  uint16_t reg0 = node.getResponseBuffer(0x00);
  uint8_t decimalPlaces = (reg0 >> 8) & 0x0F;
  uint16_t rawConcentration = node.getResponseBuffer(0x01);
  uint16_t lowAlarm = node.getResponseBuffer(0x02);
  uint16_t highAlarm = node.getResponseBuffer(0x03);
  uint16_t fullScale = node.getResponseBuffer(0x04);
  uint8_t status = node.getResponseBuffer(0x05) & 0x0F;

  float concentration = rawConcentration;
  for (int i = 0; i < decimalPlaces; i++) concentration /= 10.0;

  out += String(name) + ":" + String(concentration, 2) + " ppm";
  out += "|STATUS:" + String(status);
  out += "|LOW:" + String(lowAlarm);
  out += "|HIGH:" + String(highAlarm);
  out += "|FS:" + String(fullScale);
  return true;
}

// ==================== setup ====================
void setup() {
  // ---- UART0 用于 CH3SH ----
  Serial.begin(9600, SERIAL_8N1);
  ch3shNode.begin(CH3SH_SENSOR_ADDR, Serial);

  // ---- ADC 引脚（H2S, NH3, VOC） ----
  pinMode(H2S_ADC_PIN, INPUT);
  pinMode(NH3_ADC_PIN, INPUT);
  pinMode(VOC_ADC_PIN, INPUT);   // 新增

  // ---- 删除原 VOC 的 UART2 初始化 ----
  // Serial2 不再需要，因为 VOC 改用模拟量，因此不调用 Serial2.begin 和 vocNode.begin

  // ---- ADC 分辨率与衰减配置（可选，默认已设置） ----
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  // ---- 初始化 VOC 滤波数组 ----
  for (int i = 0; i < FILTER_WINDOW; i++) {
    filterVOC[i] = 0.0;
  }

  // ---- 蓝牙 ----
  SerialBT.begin(BT_NAME);
  SerialBT.println("# esp32_gas ready: H2S NH3 VOC(analog 500ppm) CH3SH");
}

// ==================== loop ====================
void loop() {
  unsigned long now = millis();
  if (now - lastReadTime >= READ_INTERVAL) {
    lastReadTime = now;

    // ---- 读取 H2S ----
    uint16_t h2sAdc = analogRead(H2S_ADC_PIN);
    float h2sVolt = h2sAdc * 3.3f / 4095.0f;
    float h2sVal = h2sVolt / H2S_MAX_VOLT * H2S_FULL_RANGE;

    // ---- 读取 NH3 ----
    uint16_t nh3Adc = analogRead(NH3_ADC_PIN);
    float nh3Volt = nh3Adc * 3.3f / 4095.0f;
    float nh3Val = nh3Volt / NH3_MAX_VOLT * NH3_FULL_RANGE;

    // ---- 读取 VOC（模拟量） ----
    uint16_t vocAdc = analogRead(VOC_ADC_PIN);
    float vocVolt = vocAdc * 3.3f / 4095.0f;
    if (vocVolt > VOC_MAX_VOLT) vocVolt = VOC_MAX_VOLT;
    if (vocVolt < 0) vocVolt = 0;
    float vocVal = (vocVolt / VOC_MAX_VOLT) * VOC_FULL_RANGE;

    // ---- VOC 滑动平均滤波 ----
    filterVOC[filterIndex] = vocVal;
    filterIndex = (filterIndex + 1) % FILTER_WINDOW;
    float vocAvg = 0;
    for (int i = 0; i < FILTER_WINDOW; i++) {
      vocAvg += filterVOC[i];
    }
    vocAvg /= FILTER_WINDOW;

    // ---- 构建输出字符串 ----
    String sendData =
        "H2S:" + String(h2sVal, 2) + " ppm|V:" + String(h2sVolt, 3) +
        " NH3:" + String(nh3Val, 2) + " ppm|V:" + String(nh3Volt, 3);

    // VOC 输出（浓度 + 电压）
    sendData += " VOC:" + String(vocAvg, 2) + " ppm|V:" + String(vocVolt, 3);

    // CH3SH（Modbus）
    String part;
    if (appendModbusGas(ch3shNode, CH3SH_START_REG, CH3SH_NUM_REGS, "CH3SH", part)) {
      sendData += " " + part;
    }

    sendData += "\r\n";

    if (SerialBT.hasClient()) {
      SerialBT.print(sendData);
    }
  }
}