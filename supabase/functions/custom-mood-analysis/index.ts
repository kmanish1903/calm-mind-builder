import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface MoodAnalysisRequest {
  text?: string;
  audioFeatures?: number[];
  emotionTags?: string[];
  currentScore?: number;
}

interface ModelPrediction {
  moodClass: number;
  confidence: number;
  probabilities: number[];
}

class CustomMoodAnalyzer {
  private voiceModel: any = null;
  private textModel: any = null;
  private crisisModel: any = null;

  async loadModels() {
    // In production, load ONNX models from storage
    // For now, use simplified logic
    this.voiceModel = { predict: this.mockVoicePredict };
    this.textModel = { predict: this.mockTextPredict };
    this.crisisModel = { predict: this.mockCrisisPredict };
  }

  private mockVoicePredict(features: number[]): ModelPrediction {
    // Simplified voice analysis based on feature patterns
    const energy = features[0] || 0.5;
    const pitch = features[1] || 0.5;
    
    let moodClass = 2; // default moderate
    if (energy < 0.3 && pitch < 0.4) moodClass = 1; // low
    else if (energy > 0.7 && pitch > 0.6) moodClass = 4; // excellent
    else if (energy > 0.5) moodClass = 3; // good
    
    const confidence = Math.min(0.9, Math.abs(energy - 0.5) + Math.abs(pitch - 0.5));
    const probabilities = Array(5).fill(0.1);
    probabilities[moodClass] = confidence;
    
    return { moodClass, confidence, probabilities };
  }

  private mockTextPredict(text: string): ModelPrediction {
    // Simplified text analysis
    const words = text.toLowerCase().split(' ');
    const negativeWords = ['sad', 'depressed', 'tired', 'hopeless', 'awful'];
    const positiveWords = ['happy', 'great', 'good', 'amazing', 'wonderful'];
    
    const negativeCount = words.filter(w => negativeWords.includes(w)).length;
    const positiveCount = words.filter(w => positiveWords.includes(w)).length;
    
    let moodClass = 2;
    if (negativeCount > positiveCount) moodClass = Math.max(0, 2 - negativeCount);
    else if (positiveCount > 0) moodClass = Math.min(4, 2 + positiveCount);
    
    const confidence = Math.min(0.9, (Math.abs(negativeCount - positiveCount) + 1) * 0.2);
    const probabilities = Array(5).fill(0.1);
    probabilities[moodClass] = confidence;
    
    return { moodClass, confidence, probabilities };
  }

  private mockCrisisPredict(text: string): { crisisRisk: number; riskLevel: string } {
    const crisisWords = ['kill', 'die', 'suicide', 'end it all', 'hurt myself'];
    const words = text.toLowerCase();
    
    const crisisCount = crisisWords.filter(word => words.includes(word)).length;
    const crisisRisk = Math.min(1.0, crisisCount * 0.3);
    
    let riskLevel = 'low';
    if (crisisRisk > 0.7) riskLevel = 'high';
    else if (crisisRisk > 0.4) riskLevel = 'moderate';
    
    return { crisisRisk, riskLevel };
  }

  async analyzeMultimodal(request: MoodAnalysisRequest): Promise<any> {
    const results: any = {
      analysis: '',
      concernLevel: 'low',
      suggestedMoodScore: 5,
      keywords: [],
      isCrisis: false,
      recommendations: [],
      modelPredictions: {}
    };

    // Voice analysis
    if (request.audioFeatures && request.audioFeatures.length > 0) {
      const voicePred = this.voiceModel.predict(request.audioFeatures);
      results.modelPredictions.voice = voicePred;
      results.suggestedMoodScore = Math.max(1, Math.min(10, voicePred.moodClass * 2 + 1));
    }

    // Text analysis
    if (request.text) {
      const textPred = this.textModel.predict(request.text);
      results.modelPredictions.text = textPred;
      
      // Crisis detection
      const crisisResult = this.crisisModel.predict(request.text);
      results.modelPredictions.crisis = crisisResult;
      results.isCrisis = crisisResult.riskLevel === 'high';
      
      if (results.isCrisis) {
        results.concernLevel = 'crisis';
        results.recommendations.push('Seek immediate professional help');
      }
    }

    // Fusion logic
    if (results.modelPredictions.voice && results.modelPredictions.text) {
      const voiceScore = results.modelPredictions.voice.moodClass;
      const textScore = results.modelPredictions.text.moodClass;
      const fusedScore = Math.round((voiceScore + textScore) / 2);
      results.suggestedMoodScore = Math.max(1, Math.min(10, fusedScore * 2 + 1));
    }

    // Set concern level
    if (results.suggestedMoodScore <= 3) results.concernLevel = 'high';
    else if (results.suggestedMoodScore <= 5) results.concernLevel = 'moderate';

    results.analysis = this.generateAnalysis(results);
    
    return results;
  }

  private generateAnalysis(results: any): string {
    const score = results.suggestedMoodScore;
    
    if (results.isCrisis) {
      return "I'm concerned about your wellbeing. Please reach out for immediate support.";
    } else if (score <= 3) {
      return "You seem to be going through a difficult time. Consider talking to someone you trust.";
    } else if (score <= 5) {
      return "Your mood appears moderate. Some self-care activities might help improve how you're feeling.";
    } else {
      return "You seem to be in a positive state. Keep up the good work with your mental wellness.";
    }
  }
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const analyzer = new CustomMoodAnalyzer();
    await analyzer.loadModels();
    
    const request: MoodAnalysisRequest = await req.json();
    const analysis = await analyzer.analyzeMultimodal(request);

    return new Response(
      JSON.stringify(analysis),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error("Error in custom-mood-analysis:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});