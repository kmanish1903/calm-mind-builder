import { useAuth } from "@/hooks/useAuth";
import { Navigate } from "react-router-dom";
import { ROUTES } from "@/utils/constants";
import LoadingSpinner from "./LoadingSpinner";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  const { user, session } = useAuth();

  if (session === null && user === null) {
    return <Navigate to={ROUTES.LOGIN} replace />;
  }

  if (!user) {
    return <LoadingSpinner />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;
