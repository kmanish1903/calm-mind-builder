import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "@/utils/constants";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "@/hooks/use-toast";
import { useState, useEffect } from "react";

const SettingsScreen = () => {
  const { signOut, user } = useAuth();
  const navigate = useNavigate();
  const [notificationSettings, setNotificationSettings] = useState({
    moodReminders: true,
    goalReminders: true,
    crisisFollowUp: true,
    weeklyReports: true
  });

  useEffect(() => {
    if (user) {
      loadSettings();
      requestNotificationPermission();
    }
  }, [user]);

  const loadSettings = async () => {
    if (!user) return;

    const { data } = await supabase
      .from('profiles')
      .select('preferences')
      .eq('id', user.id)
      .single();

    if (data?.preferences && typeof data.preferences === 'object') {
      const prefs = data.preferences as { notifications?: typeof notificationSettings };
      if (prefs.notifications) {
        setNotificationSettings(prefs.notifications);
      }
    }
  };

  const requestNotificationPermission = async () => {
    if ('Notification' in window && Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      if (permission === 'granted') {
        toast({
          title: "Notifications enabled",
          description: "You'll receive important reminders and updates"
        });
      }
    }
  };

  const saveSettings = async (newSettings: typeof notificationSettings) => {
    if (!user) return;

    const { data: profile } = await supabase
      .from('profiles')
      .select('preferences')
      .eq('id', user.id)
      .single();

    const existingPrefs = (profile?.preferences && typeof profile.preferences === 'object') 
      ? profile.preferences as Record<string, any>
      : {};

    const { error } = await supabase
      .from('profiles')
      .update({
        preferences: {
          ...existingPrefs,
          notifications: newSettings
        }
      })
      .eq('id', user.id);

    if (error) {
      toast({
        title: "Error",
        description: "Failed to save settings",
        variant: "destructive"
      });
    } else {
      toast({
        title: "Settings saved",
        description: "Your notification preferences have been updated"
      });
    }
  };

  const updateSetting = (key: keyof typeof notificationSettings) => {
    const newSettings = {
      ...notificationSettings,
      [key]: !notificationSettings[key]
    };
    setNotificationSettings(newSettings);
    saveSettings(newSettings);
  };

  const handleSignOut = async () => {
    await signOut();
    navigate(ROUTES.LOGIN);
  };

  const exportData = async () => {
    if (!user) return;

    const { data: moodEntries } = await supabase
      .from('mood_entries')
      .select('*')
      .eq('user_id', user.id);

    const { data: goals } = await supabase
      .from('daily_goals')
      .select('*')
      .eq('user_id', user.id);

    const { data: progress } = await supabase
      .from('user_progress')
      .select('*')
      .eq('user_id', user.id);

    const exportData = {
      moodEntries,
      goals,
      progress,
      exportedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mindcare-data-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Data exported",
      description: "Your data has been downloaded"
    });
  };

  return (
    <div className="container py-8 space-y-6">
      <h1 className="text-3xl font-bold">Settings</h1>
      
      <Card>
        <CardHeader>
          <CardTitle>Notifications</CardTitle>
          <CardDescription>Manage your notification preferences</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="mood-reminders">Daily Mood Check-in Reminders</Label>
            <Switch
              id="mood-reminders"
              checked={notificationSettings.moodReminders}
              onCheckedChange={() => updateSetting('moodReminders')}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="goal-reminders">Goal Completion Reminders</Label>
            <Switch
              id="goal-reminders"
              checked={notificationSettings.goalReminders}
              onCheckedChange={() => updateSetting('goalReminders')}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="crisis-followup">Crisis Follow-up Notifications</Label>
            <Switch
              id="crisis-followup"
              checked={notificationSettings.crisisFollowUp}
              onCheckedChange={() => updateSetting('crisisFollowUp')}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="weekly-reports">Weekly Progress Reports</Label>
            <Switch
              id="weekly-reports"
              checked={notificationSettings.weeklyReports}
              onCheckedChange={() => updateSetting('weeklyReports')}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Data & Privacy</CardTitle>
          <CardDescription>Manage your data and privacy settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button variant="outline" onClick={exportData} disabled={!user}>
            Export All Data
          </Button>
          <p className="text-sm text-muted-foreground">
            Download a complete copy of your mental health data in JSON format
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Account</CardTitle>
          <CardDescription>Manage your account settings</CardDescription>
        </CardHeader>
        <CardContent>
          <Button variant="destructive" onClick={handleSignOut}>
            Sign Out
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default SettingsScreen;
