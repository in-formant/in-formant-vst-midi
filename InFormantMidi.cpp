#include "InFormantMidi.h"
#include "IPlug_include_in_plug_src.h"
#include "IControls.h"

InFormantMidi::InFormantMidi(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  GetParam(kGain)->InitDouble("Gain", 0., 0., 100.0, 0.01, "%");
  GetParam(kInternalSampleRate)->InitFrequency("Internal sample rate", 10600.0, 6000.0, 16000.0);
  GetParam(kInternalLPOrder)->InitInt("Internal LP order", 10, 6, 16);
  GetParam(kNumFormantsOut)->InitInt("Output formant count", 2, 1, 5);

#if IPLUG_EDITOR // http://bit.ly/2S64BDd
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, GetScaleForScreen(PLUG_WIDTH, PLUG_HEIGHT));
  };
  
  mLayoutFunc = [&](IGraphics* pGraphics) {
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(COLOR_GRAY);
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    const IRECT b = pGraphics->GetBounds();
    pGraphics->AttachControl(new ITextControl(b.GetMidVPadded(50), "InFormant MIDI", IText(50)));
    pGraphics->AttachControl(new IVKnobControl(b.GetCentredInside(100).GetVShifted(-100), kGain));
  };
#endif
}

#if IPLUG_DSP
void InFormantMidi::ProcessBlock(sample** inputs, sample** outputs, const int nFrames)
{
  const double gain = GetParam(kGain)->Value() / 100.;
  const int nChans = NOutChansConnected();
  
  // Mix down to mono and to single-precision float.
  for (int s = 0; s < nFrames; ++s) {
    mMixedBlockIn[s] = float(0);
    for (int c = 0; c < nChans; ++c) {
      mMixedBlockIn[s] += float(inputs[c][s]);
    }
    mMixedBlockIn[s] /= float(nChans);
  }

  // Resample for formant estimation.
  speexport::spx_uint32_t resInLen(mMixedBlockIn.size()), resOutLen(mMixedBlockOut.size());
  mResampler.process(0, mMixedBlockIn.data(), &resInLen, mMixedBlockOut.data(), &resOutLen);

  // Push to the circular buffer.
  if (resOutLen >= mInput.size()) {
    std::copy(mMixedBlockOut.end() - mInput.size(), mMixedBlockOut.end(), mInput.begin());
  }
  else {
    std::rotate(mInput.begin(), mInput.begin() + resOutLen, mInput.end());
    std::copy(mMixedBlockOut.begin(), mMixedBlockOut.end(), mInput.begin() + resOutLen);
  }

  // Do formant estimation.
  Math::processBlock(
    mInput, mWindow, mInternalFs,
    mWork, mLpcWork, mRootsWork,
    mAllFormants, mFormants);

  // Create the MIDI messages to send.
  IMidiMsg msg;

  for (int note = 0; note < 128; ++note) {
    msg.MakeNoteOffMsg(note, 0);
    SendMidiMsg(msg);
  }

  for (auto& frequency : mFormants) {
    if (frequency < 0) continue;

    const int note = std::round(12.0 * log2(frequency / 440) + 69);

    msg.MakeNoteOnMsg(note, gain * 127, 0);
    SendMidiMsg(msg);
  }

  for (int c = 0; c < nChans; ++c) {
    std::fill(outputs[c], outputs[c] + nFrames, 0.);
  }
}

void InFormantMidi::OnReset()
{
  const double Fs = GetSampleRate();

  mInternalFs = GetParam(kInternalSampleRate)->Value();
  const int nSamples = (int)std::round(15. / 1000. * mInternalFs);

  int err;
  mResampler.init(1, Fs, mInternalFs, 5, &err);
  if (err != 0) {
    fprintf(stderr, "Resampler error: %s\n", speexport::speex_resampler_strerror(err));
  }

  mMixedBlockIn.resize(GetBlockSize(), 0.);
  mMixedBlockOut.resize((int) std::round((GetBlockSize() * mInternalFs) / Fs), 0.);
  mInput.resize(nSamples, 0.);
  mWindow = Math::makeGaussianWindow(nSamples);
  mWork.resize(nSamples, 0.);

  const int lpOrder = GetParam(kInternalLPOrder)->Value();
  const int nFormants = GetParam(kNumFormantsOut)->Value();

  mLpcWork = LpcWork{
    .b1 = std::vector<double>(1 + nSamples),
    .b2 = std::vector<double>(1 + nSamples),
    .aa = std::vector<double>(1 + lpOrder),
    .lpc = std::vector<double>(lpOrder),
  };

  mRootsWork = RootsWork{
    .P = std::vector<std::complex<double>>(lpOrder + 1),
    .P2 = std::vector<std::complex<double>>(lpOrder + 1),
    .Q = std::vector<std::complex<double>>(lpOrder + 1),
    .R = std::vector<std::complex<double>>(lpOrder),
  };

  mAllFormants.resize(lpOrder / 2);
  mFormants.resize(nFormants);
}

void InFormantMidi::OnParamChange(const int paramIdx)
{
  OnReset();
}

std::vector<float> Math::makeGaussianWindow(const int nSamples)
{
  std::vector<float> window(nSamples);
  const double edge = std::exp(-12.);
  for (int i = 0; i < nSamples; ++i) {
    const double imid = 0.5 * (nSamples + 1);
    window[i] = (std::exp(-48. * (i - imid) * (i - imid) / (nSamples + 1) / (nSamples + 1)) - edge) / (1.0 - edge);
  }
  return window;
}

static double vecBurgBuffered(double *lpc, const int m, const double *data, const int n,
  std::vector<double>& b1, std::vector<double>& b2, std::vector<double>& aa)
{
  int i, j;
  std::fill(b1.begin(), b1.end(), 0.0);
  std::fill(b2.begin(), b2.end(), 0.0);
  std::fill(aa.begin(), aa.end(), 0.0);

  double* a = &lpc[-1];
  const double* x = &data[-1];

  double p = 0.0;
  for (j = 1; j <= n; ++j)
    p += x[j] * x[j];

  double xms = p / n;
  if (xms <= 0.0) {
    return xms;
  }

  b1[1] = x[1];
  b2[n - 1] = x[n];
  for (j = 2; j <= n - 1; ++j)
    b1[j] = b2[j - 1] = x[j];

  for (i = 1; i <= m; ++i) {
    double num = 0.0, denum = 0.0;
    for (j = 1; j <= n - i; ++j) {
      num += b1[j] * b2[j];
      denum += b1[j] * b1[j] + b2[j] * b2[j];
    }

    if (denum <= 0.0)
      return 0.0;

    a[i] = 2.0 * num / denum;

    xms *= 1.0 - a[i] * a[i];

    for (j = 1; j <= i - 1; ++j)
      a[j] = aa[j] - a[i] * aa[i - j];

    if (i < m) {
      for (j = 1; j <= i; ++j)
        aa[j] = a[j];
      for (j = 1; j <= n - i - 1; ++j) {
        b1[j] -= aa[i] * b2[j];
        b2[j] = b2[j + 1] - aa[i] * b1[j + 1];
      }
    }
  }

  return xms;
}

bool Math::calculateLpc(const std::vector<double>& input, LpcWork& work)
{
  const int n = input.size();
  const int m = work.lpc.size();
  double gain = vecBurgBuffered(work.lpc.data(), m, input.data(), n, work.b1, work.b2, work.aa);
  if (gain <= 0.0) {
    return false;
  }
  gain *= n;
  std::transform(work.lpc.begin(), work.lpc.end(), work.lpc.begin(), std::negate<>());
  return true;
}

std::complex<double> Math::calculateOneRoot(const std::vector<std::complex<double>>& P, std::complex<double> xk, double accuracy)
{
  constexpr int maxIt = 1000;

  auto ys = calculatePolynomialWithDerivatives<2>(P, xk);

  const int n = P.size() - 1;

  for (int it = 0; it < maxIt; ++it) {
    if (std::abs(ys[0]) < accuracy)
      return xk;

    auto g = ys[1] / ys[0];
    auto h = g * g - ys[2] / ys[0];
    auto f = std::sqrt(((double)n - 1) * ((double)n * h - g * g));

    std::complex<double> dx;
    if (std::abs(g + f) > std::abs(g - f)) {
      dx = (double)n / (g + f);
    }
    else {
      dx = (double)n / (g - f);
    }
    xk -= dx;

    if (std::abs(dx) < accuracy)
      return xk;

    ys = calculatePolynomialWithDerivatives<2>(P, xk);
  }

  return xk;
}

void Math::deflatePolynomial(const std::vector<std::complex<double>>& P, const std::complex<double>& r, std::vector<std::complex<double>>& Q)
{
  const int n = P.size() - 1;

  Q.resize(n);
  Q[n - 1] = P[n];
  for (int i = n - 2; i >= 0; --i) {
    Q[i] = P[i + 1] + r * Q[i + 1];
  }
}

void Math::calculateRoots(RootsWork& work)
{
  const int n = work.P.size() - 1;
  work.R.resize(n);
  work.P2 = work.P;

  for (int i = 0; i < n; ++i) {
    work.R[i] = calculateOneRoot(work.P2, 0.0, 1e-6);
    deflatePolynomial(work.P2, work.R[i], work.Q);
    std::swap(work.P2, work.Q);
  }

  for (int i = 0; i < n; ++i) {
    work.R[i] = calculateOneRoot(work.P, work.R[i], 1e-12);
  }
}

void Math::processBlock(
  const std::vector<float>& input,
  const std::vector<float>& window,
  const double Fs,
  std::vector<double>& work,
  LpcWork& lpcWork,
  RootsWork& rootsWork,
  std::vector<double>& allFormants,
  std::vector<double>& formants)
{
  const int nSamples = input.size();
  const int lpOrder = lpcWork.lpc.size();

  // Copy input into work.
  for (int i = 0; i < nSamples; ++i)
    work[i] = input[i];

  // Pre-emphasis.
  const double preemphFrequency = 50.0;
  const double preemphFactor = std::exp(-(2.0 * PI * preemphFrequency) / Fs);
  for (int i = nSamples - 1; i >= 1; --i)
    work[i] -= preemphFactor * work[i - 1];

  // Windowing.
  for (int i = 0; i < nSamples; ++i)
    work[i] *= window[i];

  // Calculate its linear prediction model.
  if (!calculateLpc(work, lpcWork)) {
    std::fill(formants.begin(), formants.end(), -1);
    return;
  }

  // Transform into polynomial coefficient form.
  rootsWork.P.resize(lpOrder + 1);
  rootsWork.P[lpOrder] = 1.0;
  for (int i = 0; i < lpOrder; ++i) {
    rootsWork.P[lpOrder - 1 - i] = lpcWork.lpc[i];
  }

  // Solve for roots.
  calculateRoots(rootsWork);

  // Calculate formants from roots
  const double phiDelta = 2.0 * 50.0 * PI / Fs;

  allFormants.clear();
  for (const auto& z : rootsWork.R) {
    if (z.imag() < 0) continue;

    const double r = std::abs(z);
    const double phi = std::arg(z);

    if (r < 0.7 || r > 1.0 || phi < phiDelta || phi > PI - phiDelta) {
      continue;
    }

    const double frequency = std::abs(phi) * Fs / (2.0 * PI);

    allFormants.push_back(frequency);
  }

  std::sort(allFormants.begin(), allFormants.end(), std::less<>());

  // Only copy the first formants.
  const int nFormants = formants.size();
  for (int i = 0; i < nFormants; ++i) {
    if (i < allFormants.size())
      formants[i] = allFormants[i];
    else
      formants[i] = -1;
  }
}
#endif
