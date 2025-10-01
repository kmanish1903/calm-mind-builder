import { useState } from "react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { ROUTES } from "@/utils/constants";
import { useMood } from "@/hooks/useMood";
import MoodScale from "@/components/mood/MoodScale";
import EmotionTags from "@/components/mood/EmotionTags";
import VoiceRecorder from "@/components/mood/VoiceRecorder";
import { supabase } from "@/integrations/supabase/client";
import LoadingSpinner from "@/components/LoadingSpinner";
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const MoodCheckScreen = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const {
    currentMood,
    setCurrentMood,
    selectedEmotions,
    toggleEmotion,
    addMoodEntry,
  } = useMood();
  
  const [textNote, setTextNote] = useState("");
  const [voiceBlob, setVoiceBlob] = useState<Blob | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiInsight, setAiInsight] = useState<string | null>(null);
  const [isCrisis, setIsCrisis] = useState(false);

  const handleAnalyze = async () => {
    if (!textNote && selectedEmotions.length === 0) {
      toast({
        title: "No data to analyze",
        description: "Please add some text or select emotions first",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      const { data, error } = await supabase.functions.invoke('analyze-mood', {
        body: {
          text: textNote,
          emotionTags: selectedEmotions,
          currentScore: currentMood,
        },
      });

      if (error) throw error;

      setAiInsight(data.analysis);
      setIsCrisis(data.isCrisis);
      
      // Update mood score if AI suggests a different one
      if (data.suggestedMoodScore && Math.abs(data.suggestedMoodScore - currentMood) > 2) {
        setCurrentMood(data.suggestedMoodScore);
        toast({
          title: "Mood score adjusted",
          description: `AI analysis suggests a score of ${data.suggestedMoodScore}/10`,
        });
      }

      if (data.isCrisis) {
        toast({
          title: "Crisis Detected",
          description: "Please consider reaching out to emergency support",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Error analyzing mood:', error);
      toast({
        title: "Analysis failed",
        description: "Unable to analyze mood at this time",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSubmit = () => {
    addMoodEntry({
      moodScore: currentMood,
      emotionTags: selectedEmotions,
      textNote: textNote || undefined,
      voiceNote: voiceBlob ? "recorded" : undefined,
    });

    toast({
      title: "Mood recorded!",
      description: "Your mood has been saved successfully.",
    });

    navigate(ROUTES.DASHBOARD);
  };

  return (
    <Layout>
      <div className="container py-6 space-y-6 max-w-2xl mx-auto">
        <div>
          <h1 className="text-3xl font-bold">Mood Check-In</h1>
          <p className="text-muted-foreground">How are you feeling right now?</p>
        </div>

        {isCrisis && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Crisis Detected</AlertTitle>
            <AlertDescription>
              Based on your entry, we recommend reaching out for immediate support.
              <Button 
                variant="outline" 
                size="sm" 
                className="mt-2 w-full"
                onClick={() => navigate(ROUTES.CRISIS)}
              >
                Get Help Now
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {aiInsight && (
          <Card className="bg-primary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="text-base">AI Insight</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm">{aiInsight}</p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Rate Your Mood</CardTitle>
          </CardHeader>
          <CardContent>
            <MoodScale value={currentMood} onChange={setCurrentMood} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Select Emotions</CardTitle>
          </CardHeader>
          <CardContent>
            <EmotionTags 
              selectedEmotions={selectedEmotions}
              onToggle={toggleEmotion}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Voice Note (Optional)</CardTitle>
          </CardHeader>
          <CardContent>
            <VoiceRecorder onRecordingComplete={setVoiceBlob} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Additional Notes (Optional)</CardTitle>
          </CardHeader>
          <CardContent>
            <Textarea
              placeholder="Describe your feelings in more detail..."
              value={textNote}
              onChange={(e) => setTextNote(e.target.value)}
              rows={4}
            />
          </CardContent>
        </Card>

        <div className="flex gap-2">
          <Button 
            variant="outline" 
            onClick={() => navigate(ROUTES.DASHBOARD)}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button 
            variant="secondary"
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className="flex-1"
          >
            {isAnalyzing ? <><LoadingSpinner /> Analyzing...</> : "Analyze with AI"}
          </Button>
          <Button 
            onClick={handleSubmit}
            className="flex-1"
          >
            Save Mood Entry
          </Button>
        </div>
      </div>
    </Layout>
  );
};

export default MoodCheckScreen;
