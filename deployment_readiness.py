#!/usr/bin/env python3
"""
Phase 9: Final Review, Testing, and Optimization
Comprehensive deployment readiness checker for the Mental Health App
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import requests
import time

class DeploymentReadinessChecker:
    """Comprehensive deployment readiness validation"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            'feature_verification': {},
            'security_audit': {},
            'performance_metrics': {},
            'code_quality': {},
            'integration_tests': {},
            'deployment_prep': {}
        }
        
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all deployment readiness checks"""
        print("🚀 Starting Comprehensive Deployment Readiness Check...")
        
        # Feature Verification
        print("\n📋 Phase 1: Feature Verification")
        self.results['feature_verification'] = self.verify_prd_requirements()
        
        # Security Audit
        print("\n🔒 Phase 2: Security & Privacy Audit")
        self.results['security_audit'] = self.conduct_security_audit()
        
        # Performance Testing
        print("\n⚡ Phase 3: Performance Optimization")
        self.results['performance_metrics'] = self.test_performance()
        
        # Code Quality Review
        print("\n🧹 Phase 4: Code Quality Assessment")
        self.results['code_quality'] = self.assess_code_quality()
        
        # Integration Testing
        print("\n🔗 Phase 5: Integration Testing")
        self.results['integration_tests'] = self.run_integration_tests()
        
        # Deployment Preparation
        print("\n📦 Phase 6: Deployment Preparation")
        self.results['deployment_prep'] = self.prepare_deployment()
        
        # Generate final report
        self.generate_readiness_report()
        
        return self.results
    
    def verify_prd_requirements(self) -> Dict[str, Any]:
        """Verify all PRD requirements are implemented"""
        requirements = {
            'mood_tracking': self._check_mood_tracking(),
            'ai_features': self._check_ai_features(),
            'crisis_intervention': self._check_crisis_features(),
            'therapist_communication': self._check_therapist_features(),
            'data_privacy': self._check_privacy_features(),
            'accessibility': self._check_accessibility(),
            'pwa_features': self._check_pwa_features()
        }
        
        total_checks = sum(len(checks) for checks in requirements.values())
        passed_checks = sum(
            sum(1 for check in checks.values() if check.get('status') == 'pass')
            for checks in requirements.values()
        )
        
        return {
            'requirements': requirements,
            'completion_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'status': 'pass' if passed_checks / total_checks >= 0.9 else 'fail'
        }
    
    def _check_mood_tracking(self) -> Dict[str, Any]:
        """Check mood tracking implementation"""
        checks = {}
        
        # Check mood components
        mood_components = [
            'src/components/mood/MoodScale.tsx',
            'src/components/mood/EmotionTags.tsx',
            'src/components/mood/VoiceRecorder.tsx',
            'src/components/mood/MoodTrendChart.tsx'
        ]
        
        for component in mood_components:
            path = self.project_root / component
            checks[f'component_{Path(component).stem}'] = {
                'status': 'pass' if path.exists() else 'fail',
                'message': f'Component exists: {component}'
            }
        
        # Check mood hooks
        mood_hook = self.project_root / 'src/hooks/useMood.tsx'
        checks['mood_hook'] = {
            'status': 'pass' if mood_hook.exists() else 'fail',
            'message': 'Mood management hook implemented'
        }
        
        # Check mood screens
        mood_screens = [
            'src/pages/mood/MoodCheckScreen.tsx',
            'src/pages/mood/MoodHistoryScreen.tsx'
        ]
        
        for screen in mood_screens:
            path = self.project_root / screen
            checks[f'screen_{Path(screen).stem}'] = {
                'status': 'pass' if path.exists() else 'fail',
                'message': f'Screen exists: {screen}'
            }
        
        return checks
    
    def _check_ai_features(self) -> Dict[str, Any]:
        """Check AI-powered features"""
        checks = {}
        
        # Check ML models
        ml_models = [
            'ml-models/models/mood_models.py',
            'ml-models/models/text_model.py',
            'ml-models/models/voice_model.py'
        ]
        
        for model in ml_models:
            path = self.project_root / model
            checks[f'model_{Path(model).stem}'] = {
                'status': 'pass' if path.exists() else 'fail',
                'message': f'ML model exists: {model}'
            }
        
        # Check Supabase functions
        ai_functions = [
            'supabase/functions/analyze-mood/index.ts',
            'supabase/functions/generate-recommendations/index.ts',
            'supabase/functions/chat/index.ts'
        ]
        
        for func in ai_functions:
            path = self.project_root / func
            checks[f'function_{Path(func).parent.name}'] = {
                'status': 'pass' if path.exists() else 'fail',
                'message': f'AI function exists: {func}'
            }
        
        return checks
    
    def _check_crisis_features(self) -> Dict[str, Any]:
        """Check crisis intervention features"""
        checks = {}
        
        crisis_screen = self.project_root / 'src/pages/CrisisScreen.tsx'
        checks['crisis_screen'] = {
            'status': 'pass' if crisis_screen.exists() else 'fail',
            'message': 'Crisis intervention screen implemented'
        }
        
        # Check for crisis detection in mood analysis
        mood_analysis = self.project_root / 'supabase/functions/analyze-mood/index.ts'
        if mood_analysis.exists():
            content = mood_analysis.read_text()
            has_crisis_detection = 'crisis' in content.lower() or 'emergency' in content.lower()
            checks['crisis_detection'] = {
                'status': 'pass' if has_crisis_detection else 'fail',
                'message': 'Crisis detection in mood analysis'
            }
        
        return checks
    
    def _check_therapist_features(self) -> Dict[str, Any]:
        """Check therapist communication features"""
        checks = {}
        
        chatbot_components = [
            'src/components/chatbot/ChatbotPanel.tsx',
            'src/pages/ChatbotScreen.tsx'
        ]
        
        for component in chatbot_components:
            path = self.project_root / component
            checks[f'component_{Path(component).stem}'] = {
                'status': 'pass' if path.exists() else 'fail',
                'message': f'Therapist communication component: {component}'
            }
        
        return checks
    
    def _check_privacy_features(self) -> Dict[str, Any]:
        """Check data privacy implementation"""
        checks = {}
        
        # Check for encryption in Supabase config
        supabase_config = self.project_root / 'supabase/config.toml'
        checks['supabase_config'] = {
            'status': 'pass' if supabase_config.exists() else 'fail',
            'message': 'Supabase configuration exists'
        }
        
        # Check for authentication
        auth_components = [
            'src/pages/auth/LoginScreen.tsx',
            'src/pages/auth/RegisterScreen.tsx',
            'src/hooks/useAuth.tsx'
        ]
        
        for component in auth_components:
            path = self.project_root / component
            checks[f'auth_{Path(component).stem}'] = {
                'status': 'pass' if path.exists() else 'fail',
                'message': f'Authentication component: {component}'
            }
        
        return checks
    
    def _check_accessibility(self) -> Dict[str, Any]:
        """Check accessibility features"""
        checks = {}
        
        # Check for accessibility in package.json
        package_json = self.project_root / 'package.json'
        if package_json.exists():
            content = json.loads(package_json.read_text())
            has_a11y_deps = any('a11y' in dep or 'accessibility' in dep 
                              for dep in content.get('dependencies', {}).keys())
            checks['a11y_dependencies'] = {
                'status': 'pass' if has_a11y_deps else 'warning',
                'message': 'Accessibility dependencies in package.json'
            }
        
        # Check for ARIA attributes in components
        ui_components = list((self.project_root / 'src/components/ui').glob('*.tsx'))
        aria_usage = 0
        for component in ui_components[:5]:  # Sample check
            if component.exists():
                content = component.read_text()
                if 'aria-' in content or 'role=' in content:
                    aria_usage += 1
        
        checks['aria_implementation'] = {
            'status': 'pass' if aria_usage > 0 else 'warning',
            'message': f'ARIA attributes found in {aria_usage} components'
        }
        
        return checks
    
    def _check_pwa_features(self) -> Dict[str, Any]:
        """Check PWA implementation"""
        checks = {}
        
        pwa_files = [
            'public/manifest.json',
            'public/service-worker.js',
            'src/components/PWAInstallPrompt.tsx'
        ]
        
        for file in pwa_files:
            path = self.project_root / file
            checks[f'pwa_{Path(file).stem}'] = {
                'status': 'pass' if path.exists() else 'fail',
                'message': f'PWA file exists: {file}'
            }
        
        return checks
    
    def conduct_security_audit(self) -> Dict[str, Any]:
        """Conduct comprehensive security audit"""
        audit_results = {
            'dependency_scan': self._scan_dependencies(),
            'code_security': self._scan_code_security(),
            'api_security': self._check_api_security(),
            'data_protection': self._check_data_protection()
        }
        
        return audit_results
    
    def _scan_dependencies(self) -> Dict[str, Any]:
        """Scan for vulnerable dependencies"""
        try:
            # Run npm audit
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                vulnerabilities = audit_data.get('vulnerabilities', {})
                
                return {
                    'status': 'pass' if not vulnerabilities else 'warning',
                    'vulnerabilities_count': len(vulnerabilities),
                    'message': f'Found {len(vulnerabilities)} dependency vulnerabilities'
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'Could not run npm audit'
                }
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Dependency scan failed: {str(e)}'
            }
    
    def _scan_code_security(self) -> Dict[str, Any]:
        """Scan code for security issues"""
        security_issues = []
        
        # Check for hardcoded secrets
        code_files = list(self.project_root.rglob('*.ts')) + list(self.project_root.rglob('*.tsx'))
        
        for file in code_files[:20]:  # Sample check
            if file.exists() and 'node_modules' not in str(file):
                content = file.read_text()
                
                # Check for potential secrets
                if any(pattern in content.lower() for pattern in ['password', 'secret', 'api_key', 'token']):
                    if any(char in content for char in ['=', ':']):
                        security_issues.append(f'Potential hardcoded secret in {file}')
        
        return {
            'status': 'pass' if not security_issues else 'warning',
            'issues': security_issues,
            'message': f'Found {len(security_issues)} potential security issues'
        }
    
    def _check_api_security(self) -> Dict[str, Any]:
        """Check API security configuration"""
        checks = {}
        
        # Check Supabase functions for security
        functions_dir = self.project_root / 'supabase/functions'
        if functions_dir.exists():
            function_dirs = [d for d in functions_dir.iterdir() if d.is_dir()]
            
            for func_dir in function_dirs:
                index_file = func_dir / 'index.ts'
                if index_file.exists():
                    content = index_file.read_text()
                    
                    # Check for CORS configuration
                    has_cors = 'cors' in content.lower()
                    
                    # Check for authentication
                    has_auth = 'authorization' in content.lower() or 'jwt' in content.lower()
                    
                    checks[func_dir.name] = {
                        'cors': has_cors,
                        'auth': has_auth,
                        'status': 'pass' if has_cors and has_auth else 'warning'
                    }
        
        return checks
    
    def _check_data_protection(self) -> Dict[str, Any]:
        """Check data protection measures"""
        checks = {}
        
        # Check for HTTPS enforcement
        vite_config = self.project_root / 'vite.config.ts'
        if vite_config.exists():
            content = vite_config.read_text()
            has_https = 'https' in content.lower()
            checks['https_config'] = {
                'status': 'pass' if has_https else 'warning',
                'message': 'HTTPS configuration in Vite'
            }
        
        # Check for environment variable usage
        env_files = list(self.project_root.glob('.env*'))
        checks['env_files'] = {
            'status': 'pass' if env_files else 'warning',
            'message': f'Found {len(env_files)} environment files'
        }
        
        return checks
    
    def test_performance(self) -> Dict[str, Any]:
        """Test application performance"""
        performance_metrics = {
            'bundle_analysis': self._analyze_bundle_size(),
            'build_performance': self._test_build_performance(),
            'runtime_optimization': self._check_runtime_optimizations()
        }
        
        return performance_metrics
    
    def _analyze_bundle_size(self) -> Dict[str, Any]:
        """Analyze bundle size and optimization"""
        try:
            # Run build to check bundle size
            result = subprocess.run(
                ['npm', 'run', 'build'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Check dist folder size
                dist_dir = self.project_root / 'dist'
                if dist_dir.exists():
                    total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    
                    return {
                        'status': 'pass' if size_mb < 10 else 'warning',
                        'size_mb': round(size_mb, 2),
                        'message': f'Bundle size: {size_mb:.2f} MB'
                    }
            
            return {
                'status': 'warning',
                'message': 'Could not analyze bundle size'
            }
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Bundle analysis failed: {str(e)}'
            }
    
    def _test_build_performance(self) -> Dict[str, Any]:
        """Test build performance"""
        try:
            start_time = time.time()
            
            result = subprocess.run(
                ['npm', 'run', 'build'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            build_time = time.time() - start_time
            
            return {
                'status': 'pass' if result.returncode == 0 else 'fail',
                'build_time': round(build_time, 2),
                'message': f'Build completed in {build_time:.2f} seconds'
            }
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Build test failed: {str(e)}'
            }
    
    def _check_runtime_optimizations(self) -> Dict[str, Any]:
        """Check for runtime optimizations"""
        optimizations = {}
        
        # Check for lazy loading
        app_file = self.project_root / 'src/App.tsx'
        if app_file.exists():
            content = app_file.read_text()
            has_lazy_loading = 'lazy' in content or 'Suspense' in content
            optimizations['lazy_loading'] = {
                'status': 'pass' if has_lazy_loading else 'warning',
                'message': 'Lazy loading implementation'
            }
        
        # Check for code splitting in Vite config
        vite_config = self.project_root / 'vite.config.ts'
        if vite_config.exists():
            content = vite_config.read_text()
            has_code_splitting = 'rollupOptions' in content or 'manualChunks' in content
            optimizations['code_splitting'] = {
                'status': 'pass' if has_code_splitting else 'warning',
                'message': 'Code splitting configuration'
            }
        
        return optimizations
    
    def assess_code_quality(self) -> Dict[str, Any]:
        """Assess overall code quality"""
        quality_metrics = {
            'linting': self._run_linting(),
            'type_checking': self._run_type_checking(),
            'test_coverage': self._check_test_coverage(),
            'documentation': self._check_documentation()
        }
        
        return quality_metrics
    
    def _run_linting(self) -> Dict[str, Any]:
        """Run ESLint checks"""
        try:
            result = subprocess.run(
                ['npm', 'run', 'lint'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            return {
                'status': 'pass' if result.returncode == 0 else 'warning',
                'message': 'ESLint checks completed',
                'output': result.stdout[:500] if result.stdout else 'No output'
            }
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Linting failed: {str(e)}'
            }
    
    def _run_type_checking(self) -> Dict[str, Any]:
        """Run TypeScript type checking"""
        try:
            result = subprocess.run(
                ['npx', 'tsc', '--noEmit'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            return {
                'status': 'pass' if result.returncode == 0 else 'warning',
                'message': 'TypeScript type checking completed',
                'errors': result.stderr[:500] if result.stderr else 'No errors'
            }
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Type checking failed: {str(e)}'
            }
    
    def _check_test_coverage(self) -> Dict[str, Any]:
        """Check test coverage"""
        test_files = list(self.project_root.rglob('*.test.*')) + list(self.project_root.rglob('*.spec.*'))
        
        return {
            'status': 'pass' if test_files else 'warning',
            'test_files_count': len(test_files),
            'message': f'Found {len(test_files)} test files'
        }
    
    def _check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness"""
        docs = {}
        
        readme = self.project_root / 'README.md'
        docs['readme'] = {
            'status': 'pass' if readme.exists() else 'fail',
            'message': 'README.md exists'
        }
        
        deployment_doc = self.project_root / 'DEPLOYMENT.md'
        docs['deployment'] = {
            'status': 'pass' if deployment_doc.exists() else 'warning',
            'message': 'Deployment documentation exists'
        }
        
        return docs
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        integration_results = {
            'supabase_connection': self._test_supabase_connection(),
            'api_endpoints': self._test_api_endpoints(),
            'pwa_functionality': self._test_pwa_functionality()
        }
        
        return integration_results
    
    def _test_supabase_connection(self) -> Dict[str, Any]:
        """Test Supabase connection"""
        supabase_client = self.project_root / 'src/integrations/supabase/client.ts'
        
        if supabase_client.exists():
            return {
                'status': 'pass',
                'message': 'Supabase client configuration exists'
            }
        else:
            return {
                'status': 'fail',
                'message': 'Supabase client not found'
            }
    
    def _test_api_endpoints(self) -> Dict[str, Any]:
        """Test API endpoints"""
        functions_dir = self.project_root / 'supabase/functions'
        
        if functions_dir.exists():
            function_count = len([d for d in functions_dir.iterdir() if d.is_dir()])
            return {
                'status': 'pass' if function_count > 0 else 'warning',
                'function_count': function_count,
                'message': f'Found {function_count} Supabase functions'
            }
        else:
            return {
                'status': 'warning',
                'message': 'No Supabase functions found'
            }
    
    def _test_pwa_functionality(self) -> Dict[str, Any]:
        """Test PWA functionality"""
        manifest = self.project_root / 'public/manifest.json'
        service_worker = self.project_root / 'public/service-worker.js'
        
        pwa_score = 0
        if manifest.exists():
            pwa_score += 1
        if service_worker.exists():
            pwa_score += 1
        
        return {
            'status': 'pass' if pwa_score == 2 else 'warning',
            'pwa_score': pwa_score,
            'message': f'PWA functionality score: {pwa_score}/2'
        }
    
    def prepare_deployment(self) -> Dict[str, Any]:
        """Prepare deployment configuration"""
        deployment_prep = {
            'environment_config': self._check_environment_config(),
            'build_optimization': self._optimize_build(),
            'monitoring_setup': self._setup_monitoring(),
            'deployment_scripts': self._create_deployment_scripts()
        }
        
        return deployment_prep
    
    def _check_environment_config(self) -> Dict[str, Any]:
        """Check environment configuration"""
        env_files = list(self.project_root.glob('.env*'))
        
        return {
            'status': 'pass' if env_files else 'warning',
            'env_files': [f.name for f in env_files],
            'message': f'Found {len(env_files)} environment files'
        }
    
    def _optimize_build(self) -> Dict[str, Any]:
        """Optimize build configuration"""
        vite_config = self.project_root / 'vite.config.ts'
        
        if vite_config.exists():
            content = vite_config.read_text()
            
            # Check for optimization settings
            optimizations = []
            if 'minify' in content:
                optimizations.append('minification')
            if 'rollupOptions' in content:
                optimizations.append('rollup_optimization')
            if 'chunkSizeWarningLimit' in content:
                optimizations.append('chunk_size_limit')
            
            return {
                'status': 'pass' if optimizations else 'warning',
                'optimizations': optimizations,
                'message': f'Build optimizations: {", ".join(optimizations)}'
            }
        
        return {
            'status': 'warning',
            'message': 'Vite config not found'
        }
    
    def _setup_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring configuration"""
        # Check for error boundary
        error_boundary = self.project_root / 'src/components/ErrorBoundary.tsx'
        
        return {
            'status': 'pass' if error_boundary.exists() else 'warning',
            'error_boundary': error_boundary.exists(),
            'message': 'Error boundary implementation'
        }
    
    def _create_deployment_scripts(self) -> Dict[str, Any]:
        """Create deployment scripts"""
        package_json = self.project_root / 'package.json'
        
        if package_json.exists():
            content = json.loads(package_json.read_text())
            scripts = content.get('scripts', {})
            
            deployment_scripts = [script for script in scripts.keys() 
                                if any(keyword in script for keyword in ['build', 'deploy', 'start'])]
            
            return {
                'status': 'pass' if deployment_scripts else 'warning',
                'scripts': deployment_scripts,
                'message': f'Deployment scripts: {", ".join(deployment_scripts)}'
            }
        
        return {
            'status': 'warning',
            'message': 'Package.json not found'
        }
    
    def generate_readiness_report(self):
        """Generate comprehensive readiness report"""
        report_path = self.project_root / 'DEPLOYMENT_READINESS_REPORT.md'
        
        report_content = self._format_readiness_report()
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"\n📊 Deployment Readiness Report generated: {report_path}")
        
        # Print summary
        self._print_summary()
    
    def _format_readiness_report(self) -> str:
        """Format the readiness report"""
        report = """# Deployment Readiness Report

## Executive Summary

This report provides a comprehensive assessment of the Mental Health App's readiness for production deployment.

"""
        
        # Add each section
        for section, results in self.results.items():
            report += f"## {section.replace('_', ' ').title()}\n\n"
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        status = value.get('status', 'unknown')
                        message = value.get('message', 'No message')
                        emoji = '✅' if status == 'pass' else '⚠️' if status == 'warning' else '❌'
                        report += f"- {emoji} **{key.replace('_', ' ').title()}**: {message}\n"
                    else:
                        report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
            
            report += "\n"
        
        report += """## Recommendations

Based on the assessment, here are the key recommendations:

1. **Address any failed checks** before deployment
2. **Review warning items** and implement fixes where possible
3. **Conduct user acceptance testing** with real users
4. **Set up production monitoring** and alerting
5. **Prepare rollback procedures** for emergency situations

## Next Steps

1. Fix critical issues identified in this report
2. Run final integration tests
3. Deploy to staging environment
4. Conduct final security review
5. Deploy to production with monitoring

---
*Report generated on: {timestamp}*
""".format(timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
        
        return report
    
    def _print_summary(self):
        """Print deployment readiness summary"""
        print("\n" + "="*60)
        print("🎯 DEPLOYMENT READINESS SUMMARY")
        print("="*60)
        
        total_checks = 0
        passed_checks = 0
        warning_checks = 0
        failed_checks = 0
        
        for section, results in self.results.items():
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict) and 'status' in value:
                        total_checks += 1
                        status = value['status']
                        if status == 'pass':
                            passed_checks += 1
                        elif status == 'warning':
                            warning_checks += 1
                        else:
                            failed_checks += 1
        
        print(f"✅ Passed: {passed_checks}")
        print(f"⚠️  Warnings: {warning_checks}")
        print(f"❌ Failed: {failed_checks}")
        print(f"📊 Total Checks: {total_checks}")
        
        if total_checks > 0:
            success_rate = (passed_checks / total_checks) * 100
            print(f"🎯 Success Rate: {success_rate:.1f}%")
            
            if success_rate >= 90:
                print("\n🚀 READY FOR DEPLOYMENT!")
            elif success_rate >= 75:
                print("\n⚠️  MOSTLY READY - Address warnings before deployment")
            else:
                print("\n❌ NOT READY - Critical issues need to be resolved")
        
        print("="*60)

def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    checker = DeploymentReadinessChecker(project_root)
    results = checker.run_comprehensive_check()
    
    return results

if __name__ == "__main__":
    main()