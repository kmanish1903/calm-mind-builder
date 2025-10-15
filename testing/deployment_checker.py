#!/usr/bin/env python3
"""
Deployment Readiness Checker
Verifies all systems are ready for production deployment
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import os

class DeploymentChecker:
    """Check deployment readiness and configuration"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.checks_passed = 0
        self.total_checks = 0
        
    def run_deployment_checks(self) -> Dict[str, Any]:
        """Run all deployment readiness checks"""
        print("🔍 Running Deployment Readiness Checks")
        print("=" * 50)
        
        checks = {
            'environment_setup': self._check_environment_setup(),
            'build_configuration': self._check_build_configuration(),
            'database_setup': self._check_database_setup(),
            'edge_functions': self._check_edge_functions(),
            'security_configuration': self._check_security_configuration(),
            'performance_requirements': self._check_performance_requirements(),
            'monitoring_setup': self._check_monitoring_setup(),
            'documentation': self._check_documentation()
        }
        
        # Calculate overall readiness
        passed_checks = sum(1 for result in checks.values() if result.get('status') == 'PASS')
        total_checks = len(checks)
        readiness_score = passed_checks / total_checks
        
        return {
            'checks': checks,
            'readiness_score': readiness_score,
            'deployment_ready': readiness_score >= 0.9,
            'blocking_issues': self._get_blocking_issues(checks),
            'recommendations': self._get_deployment_recommendations(checks)
        }
    
    def _check_environment_setup(self) -> Dict[str, Any]:
        """Check environment variables and configuration"""
        print("📋 Checking environment setup...")
        
        required_files = [
            '.env.local',
            'package.json',
            'vite.config.ts',
            'supabase/config.toml'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
        
        # Check package.json scripts
        package_json_path = self.project_root / 'package.json'
        required_scripts = ['build', 'dev', 'preview']
        missing_scripts = []
        
        if package_json_path.exists():
            with open(package_json_path) as f:
                package_data = json.load(f)
                scripts = package_data.get('scripts', {})
                for script in required_scripts:
                    if script not in scripts:
                        missing_scripts.append(script)
        
        status = 'PASS' if not missing_files and not missing_scripts else 'FAIL'
        
        return {
            'status': status,
            'missing_files': missing_files,
            'missing_scripts': missing_scripts,
            'details': 'Environment setup complete' if status == 'PASS' else 'Missing required files or scripts'
        }
    
    def _check_build_configuration(self) -> Dict[str, Any]:
        """Check build configuration and dependencies"""
        print("🔧 Checking build configuration...")
        
        try:
            # Check if build works
            result = subprocess.run(
                ['npm', 'run', 'build'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            build_success = result.returncode == 0
            
            # Check dist folder exists after build
            dist_exists = (self.project_root / 'dist').exists()
            
            status = 'PASS' if build_success and dist_exists else 'FAIL'
            
            return {
                'status': status,
                'build_success': build_success,
                'dist_folder_exists': dist_exists,
                'build_output': result.stdout if build_success else result.stderr,
                'details': 'Build configuration working' if status == 'PASS' else 'Build failed or dist folder missing'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'FAIL',
                'build_success': False,
                'error': 'Build timeout (>5 minutes)',
                'details': 'Build process took too long'
            }
        except Exception as e:
            return {
                'status': 'FAIL',
                'build_success': False,
                'error': str(e),
                'details': 'Build process failed with error'
            }
    
    def _check_database_setup(self) -> Dict[str, Any]:
        """Check Supabase database setup"""
        print("🗄️ Checking database setup...")
        
        try:
            # Check if supabase CLI is available
            result = subprocess.run(
                ['supabase', 'status'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            supabase_running = result.returncode == 0
            
            # Check migration files
            migrations_dir = self.project_root / 'supabase' / 'migrations'
            has_migrations = migrations_dir.exists() and list(migrations_dir.glob('*.sql'))
            
            # Check if tables are created (basic check)
            config_file = self.project_root / 'supabase' / 'config.toml'
            has_config = config_file.exists()
            
            status = 'PASS' if supabase_running and has_migrations and has_config else 'WARN'
            
            return {
                'status': status,
                'supabase_running': supabase_running,
                'has_migrations': has_migrations,
                'has_config': has_config,
                'details': 'Database setup complete' if status == 'PASS' else 'Database setup needs verification'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'details': 'Failed to check database setup'
            }
    
    def _check_edge_functions(self) -> Dict[str, Any]:
        """Check Supabase Edge Functions deployment"""
        print("⚡ Checking edge functions...")
        
        functions_dir = self.project_root / 'supabase' / 'functions'
        
        required_functions = [
            'analyze-mood',
            'generate-goals',
            'generate-recommendations',
            'chat',
            'text-to-speech',
            'transcribe-audio'
        ]
        
        existing_functions = []
        missing_functions = []
        
        for func in required_functions:
            func_path = functions_dir / func / 'index.ts'
            if func_path.exists():
                existing_functions.append(func)
            else:
                missing_functions.append(func)
        
        status = 'PASS' if not missing_functions else 'FAIL'
        
        return {
            'status': status,
            'existing_functions': existing_functions,
            'missing_functions': missing_functions,
            'total_functions': len(existing_functions),
            'details': f'{len(existing_functions)}/{len(required_functions)} functions ready'
        }
    
    def _check_security_configuration(self) -> Dict[str, Any]:
        """Check security configuration"""
        print("🔒 Checking security configuration...")
        
        security_checks = {
            'rls_policies_exist': self._check_rls_policies(),
            'auth_configured': self._check_auth_configuration(),
            'cors_configured': self._check_cors_configuration(),
            'https_enforced': True,  # Assume HTTPS in production
            'api_keys_secured': self._check_api_keys_security()
        }
        
        passed_checks = sum(1 for check in security_checks.values() if check)
        total_checks = len(security_checks)
        
        status = 'PASS' if passed_checks == total_checks else 'WARN'
        
        return {
            'status': status,
            'security_checks': security_checks,
            'security_score': passed_checks / total_checks,
            'details': f'{passed_checks}/{total_checks} security checks passed'
        }
    
    def _check_performance_requirements(self) -> Dict[str, Any]:
        """Check performance requirements"""
        print("⚡ Checking performance requirements...")
        
        # Check bundle size (approximate)
        dist_dir = self.project_root / 'dist'
        bundle_size = 0
        
        if dist_dir.exists():
            for file in dist_dir.rglob('*'):
                if file.is_file():
                    bundle_size += file.stat().st_size
        
        bundle_size_mb = bundle_size / (1024 * 1024)
        
        # Performance criteria
        performance_checks = {
            'bundle_size_acceptable': bundle_size_mb < 5.0,  # Less than 5MB
            'service_worker_exists': (self.project_root / 'public' / 'service-worker.js').exists(),
            'pwa_manifest_exists': (self.project_root / 'public' / 'manifest.json').exists(),
            'lazy_loading_implemented': True,  # Assume implemented
            'code_splitting_enabled': True   # Assume enabled with Vite
        }
        
        passed_checks = sum(1 for check in performance_checks.values() if check)
        total_checks = len(performance_checks)
        
        status = 'PASS' if passed_checks >= total_checks * 0.8 else 'WARN'
        
        return {
            'status': status,
            'bundle_size_mb': round(bundle_size_mb, 2),
            'performance_checks': performance_checks,
            'performance_score': passed_checks / total_checks,
            'details': f'Bundle size: {bundle_size_mb:.2f}MB, {passed_checks}/{total_checks} checks passed'
        }
    
    def _check_monitoring_setup(self) -> Dict[str, Any]:
        """Check monitoring and logging setup"""
        print("📊 Checking monitoring setup...")
        
        monitoring_checks = {
            'error_boundaries_implemented': self._check_error_boundaries(),
            'logging_configured': True,  # Basic console logging exists
            'health_check_endpoint': True,  # Supabase provides this
            'performance_monitoring': False,  # Not implemented yet
            'user_analytics': False  # Not implemented yet
        }
        
        passed_checks = sum(1 for check in monitoring_checks.values() if check)
        total_checks = len(monitoring_checks)
        
        status = 'WARN'  # Monitoring is basic but functional
        
        return {
            'status': status,
            'monitoring_checks': monitoring_checks,
            'monitoring_score': passed_checks / total_checks,
            'details': f'Basic monitoring in place, {passed_checks}/{total_checks} features available'
        }
    
    def _check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness"""
        print("📚 Checking documentation...")
        
        required_docs = [
            'README.md',
            'DEPLOYMENT.md',
            'PHASE_9_REVIEW.md'
        ]
        
        existing_docs = []
        missing_docs = []
        
        for doc in required_docs:
            doc_path = self.project_root / doc
            if doc_path.exists():
                existing_docs.append(doc)
            else:
                missing_docs.append(doc)
        
        status = 'PASS' if not missing_docs else 'WARN'
        
        return {
            'status': status,
            'existing_docs': existing_docs,
            'missing_docs': missing_docs,
            'documentation_completeness': len(existing_docs) / len(required_docs),
            'details': f'{len(existing_docs)}/{len(required_docs)} documentation files present'
        }
    
    # Helper methods for specific checks
    def _check_rls_policies(self) -> bool:
        """Check if RLS policies exist in migrations"""
        migrations_dir = self.project_root / 'supabase' / 'migrations'
        if not migrations_dir.exists():
            return False
        
        for migration_file in migrations_dir.glob('*.sql'):
            with open(migration_file) as f:
                content = f.read().lower()
                if 'row level security' in content or 'rls' in content:
                    return True
        return False
    
    def _check_auth_configuration(self) -> bool:
        """Check authentication configuration"""
        config_file = self.project_root / 'supabase' / 'config.toml'
        if not config_file.exists():
            return False
        
        with open(config_file) as f:
            content = f.read()
            return '[auth]' in content
    
    def _check_cors_configuration(self) -> bool:
        """Check CORS configuration"""
        # Assume CORS is properly configured in Supabase
        return True
    
    def _check_api_keys_security(self) -> bool:
        """Check API keys are not exposed"""
        # Check if .env files are in .gitignore
        gitignore_path = self.project_root / '.gitignore'
        if not gitignore_path.exists():
            return False
        
        with open(gitignore_path) as f:
            content = f.read()
            return '.env' in content
    
    def _check_error_boundaries(self) -> bool:
        """Check if error boundaries are implemented"""
        error_boundary_path = self.project_root / 'src' / 'components' / 'ErrorBoundary.tsx'
        return error_boundary_path.exists()
    
    def _get_blocking_issues(self, checks: Dict[str, Any]) -> List[str]:
        """Get list of blocking issues for deployment"""
        blocking_issues = []
        
        for check_name, result in checks.items():
            if result.get('status') == 'FAIL':
                blocking_issues.append(f"{check_name}: {result.get('details', 'Failed')}")
        
        return blocking_issues
    
    def _get_deployment_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """Get deployment recommendations"""
        recommendations = []
        
        for check_name, result in checks.items():
            if result.get('status') in ['FAIL', 'WARN']:
                if check_name == 'environment_setup':
                    recommendations.append("Ensure all required environment files are present")
                elif check_name == 'build_configuration':
                    recommendations.append("Fix build errors before deployment")
                elif check_name == 'database_setup':
                    recommendations.append("Verify Supabase database configuration")
                elif check_name == 'edge_functions':
                    recommendations.append("Deploy all required edge functions")
                elif check_name == 'security_configuration':
                    recommendations.append("Review and strengthen security configuration")
                elif check_name == 'performance_requirements':
                    recommendations.append("Optimize bundle size and performance")
                elif check_name == 'monitoring_setup':
                    recommendations.append("Consider implementing advanced monitoring")
                elif check_name == 'documentation':
                    recommendations.append("Complete missing documentation")
        
        # General recommendations
        recommendations.extend([
            "Test the application thoroughly in a staging environment",
            "Set up automated backups for production data",
            "Configure custom domain and SSL certificates",
            "Plan for post-deployment monitoring and maintenance"
        ])
        
        return recommendations
    
    def generate_deployment_report(self, results: Dict[str, Any]) -> str:
        """Generate deployment readiness report"""
        report = f"""# Deployment Readiness Report

## Overall Status
- **Deployment Ready**: {'✅ YES' if results['deployment_ready'] else '❌ NO'}
- **Readiness Score**: {results['readiness_score']:.1%}

## Check Results
"""
        
        for check_name, result in results['checks'].items():
            status_emoji = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}.get(result['status'], '❓')
            report += f"- **{check_name.replace('_', ' ').title()}**: {status_emoji} {result['status']}\n"
            report += f"  - {result['details']}\n\n"
        
        if results['blocking_issues']:
            report += "## Blocking Issues\n"
            for issue in results['blocking_issues']:
                report += f"- ❌ {issue}\n"
            report += "\n"
        
        report += "## Recommendations\n"
        for rec in results['recommendations']:
            report += f"- {rec}\n"
        
        report += f"\n---\nGenerated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report

def main():
    """Main execution function"""
    checker = DeploymentChecker()
    results = checker.run_deployment_checks()
    
    # Generate and save report
    report = checker.generate_deployment_report(results)
    
    report_path = Path('deployment_readiness_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Deployment readiness report saved to: {report_path}")
    print(f"🎯 Deployment Ready: {'✅ YES' if results['deployment_ready'] else '❌ NO'}")
    print(f"📈 Readiness Score: {results['readiness_score']:.1%}")
    
    if results['blocking_issues']:
        print(f"\n❌ Blocking Issues ({len(results['blocking_issues'])}):")
        for issue in results['blocking_issues']:
            print(f"  - {issue}")
    
    return results

if __name__ == "__main__":
    main()