import { Bell, Settings, AlertCircle, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "@/utils/constants";

const Header = () => {
  const navigate = useNavigate();

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-bold text-primary">MindCare AI</h1>
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate(ROUTES.CRISIS)}
            className="text-destructive hover:text-destructive hover:bg-destructive/10"
            aria-label="Crisis Support"
          >
            <AlertCircle className="h-5 w-5" />
          </Button>
          
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate(ROUTES.CHATBOT)}
            aria-label="AI Chatbot"
          >
            <MessageSquare className="h-5 w-5" />
          </Button>
          
          <Button
            variant="ghost"
            size="icon"
            aria-label="Notifications"
          >
            <Bell className="h-5 w-5" />
          </Button>
          
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate(ROUTES.SETTINGS.MAIN)}
            aria-label="Settings"
          >
            <Settings className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;
