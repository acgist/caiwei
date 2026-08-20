/**
 * 语音转写
 * 
 * - ANS
 * - VAD
 * - AGC
 * 
 *  ## 语音转写完整处理流程
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
 * |AEC|Acoustic Echo Cancellation|回声消除|
 * |ANS|Acoustic Noise Suppression|噪声抑制、降噪|
 * |VAD|Voice Activity Detection|语音活动检测、语音端点检测、静音检测|
 * |AGC|Automatic Gain Control|自动增益控制、音量自动调节|
 * |ASR|Automatic Speech Recognition|自动语音识别、语音转写|
 * |Diar|Speaker Diarization|说话人分簇、说话人日志|
 * |SID|Speaker Identification|说话人识别、声纹识别|
 */

// ANSModel
// VADModel
// AGCModel
// ASRModel: ANS -> VAD -> AGC -> ASR
