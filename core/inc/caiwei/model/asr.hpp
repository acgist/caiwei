/**
 * 自动语音识别模型
 * 
 * - ANS
 * - VAD
 * - AGC
 * 
 *  ## 完整语音处理流程
 * 
 * - AEC
 * - ANS
 * - VAD
 * - AGC
 * - ASR
 * - Diar
 * - SID
 * 
 * |英文缩写|英文全称|中文术语|
 * |:--|:--|:--|
 * |AEC |Acoustic Echo Cancellation  |回声消除|
 * |ANS |Acoustic Noise Suppression  |噪声抑制、降噪|
 * |VAD |Voice Activity Detection    |语音活动检测、语音端点检测、静音检测|
 * |AGC |Automatic Gain Control      |自动增益控制、音量自动调节|
 * |ASR |Automatic Speech Recognition|自动语音识别、语音转写|
 * |Diar|Speaker Diarization         |说话人分簇、说话人日志|
 * |SID |Speaker Identification      |说话人识别、声纹识别|
 */
#ifndef CAIWEI_MODEL_ASR_HPP
#define CAIWEI_MODEL_ASR_HPP

namespace caiwei {
namespace model  {

/**
 * 噪声抑制模型
 */
class ASNModel {};

/**
 * 语音活动检测模型
 */
class VADModel {};

/**
 * 自动增益控制模型
 */
class AGCModel {};

/**
 * 自动语音识别模型
 * 
 * 执行流程：ANS -> VAD -> AGC -> ASR
 */
class ASRModel {};

} // namespace model
} // namespace caiwei

#endif // CAIWEI_MODEL_ASR_HPP
