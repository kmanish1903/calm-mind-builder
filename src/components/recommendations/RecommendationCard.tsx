import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ExternalLink, Music, Video } from "lucide-react";

interface RecommendationCardProps {
  title: string;
  description: string;
  source: 'youtube' | 'spotify' | 'article';
  url: string;
  thumbnail?: string;
}

const RecommendationCard = ({ title, description, source, url, thumbnail }: RecommendationCardProps) => {
  const getIcon = () => {
    switch (source) {
      case 'youtube':
        return <Video className="h-4 w-4" />;
      case 'spotify':
        return <Music className="h-4 w-4" />;
      default:
        return <ExternalLink className="h-4 w-4" />;
    }
  };

  return (
    <Card className="w-full hover:shadow-lg transition-all duration-300 hover:scale-[1.02] animate-in fade-in slide-in-from-bottom-2">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            <CardTitle className="text-base line-clamp-2">{title}</CardTitle>
            {description && (
              <CardDescription className="text-sm mt-1 line-clamp-2">
                {description}
              </CardDescription>
            )}
          </div>
          {thumbnail && (
            <img 
              src={thumbnail} 
              alt={title}
              className="w-20 h-20 object-cover rounded transition-transform hover:scale-105"
              loading="lazy"
            />
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Button
          variant="outline"
          size="sm"
          className="w-full transition-all hover:bg-primary hover:text-primary-foreground"
          onClick={() => window.open(url, '_blank', 'noopener,noreferrer')}
        >
          {getIcon()}
          <span className="ml-2">Open {source}</span>
        </Button>
      </CardContent>
    </Card>
  );
};

export default RecommendationCard;
