import Layout from "@/components/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Heart, Target, TrendingUp, Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "@/utils/constants";
import { useGoals } from "@/hooks/useGoals";
import { useMood } from "@/hooks/useMood";

const DashboardScreen = () => {
  const navigate = useNavigate();
  const { goals, getCompletionRate } = useGoals();
  const { moodHistory, getMoodLabel, getMoodColor } = useMood();

  const todaysGoals = goals.slice(0, 3);
  const completionRate = getCompletionRate();
  const latestMood = moodHistory[0];

  return (
    <Layout>
      <div className="container py-6 space-y-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold">Welcome Back</h1>
          <p className="text-muted-foreground">Here's your mental wellness overview</p>
        </div>

        <Card className="bg-gradient-to-br from-primary/10 to-accent/10 border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Heart className="h-5 w-5" />
              Current Mood
            </CardTitle>
            <CardDescription>
              {latestMood 
                ? `You're feeling ${getMoodLabel(latestMood.moodScore).toLowerCase()}`
                : "How are you feeling today?"
              }
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {latestMood && (
              <div className="text-center py-4">
                <div className="text-5xl mb-2">
                  {latestMood.moodScore <= 2 && "😢"}
                  {latestMood.moodScore > 2 && latestMood.moodScore <= 4 && "😕"}
                  {latestMood.moodScore > 4 && latestMood.moodScore <= 6 && "😐"}
                  {latestMood.moodScore > 6 && latestMood.moodScore <= 8 && "🙂"}
                  {latestMood.moodScore > 8 && "😊"}
                </div>
                <div className="text-2xl font-bold" style={{ color: getMoodColor(latestMood.moodScore) }}>
                  {latestMood.moodScore}/10
                </div>
              </div>
            )}
            <Button 
              onClick={() => navigate(ROUTES.MOOD.CHECK)}
              className="w-full"
            >
              <Heart className="mr-2 h-4 w-4" />
              {latestMood ? 'Update Your Mood' : 'Check Your Mood'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Today's Goals
            </CardTitle>
            <CardDescription>
              {completionRate}% complete
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={completionRate} className="h-2" />
            <div className="space-y-2">
              {todaysGoals.map((goal) => (
                <div key={goal.id} className="flex items-center gap-2 text-sm">
                  <div className={`w-2 h-2 rounded-full ${goal.completed ? 'bg-primary' : 'bg-muted'}`} />
                  <span className={goal.completed ? 'line-through text-muted-foreground' : ''}>
                    {goal.title}
                  </span>
                </div>
              ))}
            </div>
            <Button 
              onClick={() => navigate(ROUTES.GOALS.LIST)}
              variant="outline"
              className="w-full"
            >
              <Target className="mr-2 h-4 w-4" />
              View All Goals
            </Button>
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-2">
          <Card className="cursor-pointer hover:border-primary transition-colors" onClick={() => navigate(ROUTES.PROGRESS)}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <TrendingUp className="h-4 w-4" />
                Progress
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">View your mental health trends</p>
            </CardContent>
          </Card>

          <Card className="cursor-pointer hover:border-primary transition-colors" onClick={() => navigate(ROUTES.RECOMMENDATIONS.HUB)}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Sparkles className="h-4 w-4" />
                Recommendations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">Personalized wellness activities</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
};

export default DashboardScreen;
