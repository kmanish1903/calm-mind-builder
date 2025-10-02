# MindCare AI - Deployment Guide

## Quick Start Deployment

### 1. Pre-Deployment Checklist

**Supabase Configuration** (REQUIRED):
- [ ] Enable leaked password protection: Supabase Dashboard → Authentication → Providers → Email → Enable "Leaked Password Protection"
- [ ] Set Site URL: Supabase Dashboard → Authentication → URL Configuration → Site URL
- [ ] Set Redirect URLs: Add your production domain to Authorized Redirect URLs
- [ ] Verify all edge functions are deployed (they deploy automatically with Lovable)

**Legal & Compliance** (REQUIRED for HIPAA):
- [ ] Create Privacy Policy
- [ ] Create Terms of Service
- [ ] Add HIPAA compliance documentation
- [ ] Ensure BAA (Business Associate Agreement) with Supabase if handling PHI

---

## 2. Vercel Deployment

MindCare AI can be deployed directly from Lovable:

1. Click the **Publish** button in the top right of the Lovable editor
2. Your app will be deployed to a Lovable subdomain: `your-project.lovable.app`
3. For custom domains, go to Project → Settings → Domains

---

## 3. Environment Configuration

### Supabase Secrets (Already Configured)
The following secrets are pre-configured in your Supabase project:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `LOVABLE_API_KEY`
- `SUPABASE_DB_URL`
- `SUPABASE_PUBLISHABLE_KEY`

**No additional configuration needed** - edge functions automatically have access to these secrets.

---

## 4. Database Migrations

All database migrations are already applied:
- ✅ `profiles` table with RLS policies
- ✅ `mood_entries` table with RLS policies
- ✅ `daily_goals` table with RLS policies
- ✅ `user_progress` table with RLS policies
- ✅ `crisis_events` table with RLS policies
- ✅ Triggers for `updated_at` timestamps
- ✅ Trigger for automatic profile creation on user signup

**Verification**: Check Supabase Dashboard → Table Editor to confirm all tables exist.

---

## 5. Edge Functions

The following edge functions are automatically deployed:

1. **analyze-mood**: AI-powered mood analysis using Google Gemini 2.5 Flash
2. **generate-goals**: Personalized daily goal generation
3. **generate-recommendations**: Mood-based recommendations

**Verification**: 
- Check Supabase Dashboard → Edge Functions
- Test each function through the app interface

---

## 6. Authentication Setup

### Email/Password Authentication (Configured)
Already enabled with:
- Session persistence
- Auto-refresh tokens
- Secure password storage

### Required Configuration
1. **Site URL**: Set to your production domain
   - Navigate to: Supabase Dashboard → Authentication → URL Configuration
   - Example: `https://yourdomain.com`

2. **Redirect URLs**: Add all deployment URLs
   - Development: `http://localhost:8080`
   - Staging: `https://your-project.lovable.app`
   - Production: `https://yourdomain.com`

3. **Email Templates**: Customize in Supabase Dashboard → Authentication → Email Templates
   - Confirmation email
   - Password reset email
   - Magic link email (if enabled)

---

## 7. PWA Configuration

### Service Worker
The service worker is automatically registered in `src/main.tsx` and will cache:
- Core app files
- Static assets
- API responses (when appropriate)

### Manifest
PWA manifest is located at `public/manifest.json` with:
- App name: MindCare AI
- Theme color: #6B73FF
- Display mode: standalone
- Icons: Currently using placeholder (update with actual app icons)

**Production TODO**:
1. Replace placeholder icons in `public/manifest.json` with actual app icons
2. Generate required icon sizes: 192x192, 512x512
3. Test PWA installation on mobile devices

---

## 8. Security Verification

### Pre-Production Security Checklist
- [x] Row Level Security enabled on all tables
- [x] User data isolation verified
- [x] Edge functions use authentication
- [x] No sensitive data in client code
- [x] CORS headers configured
- [ ] Leaked password protection enabled (user action required)
- [ ] SSL/TLS certificates (automatic with Lovable/Supabase)

### Testing Security
1. Test with different user accounts to verify data isolation
2. Attempt unauthorized API calls (should fail)
3. Verify crisis event logging works
4. Test data export functionality

---

## 9. Monitoring & Maintenance

### Edge Function Logs
Monitor edge function performance:
- Supabase Dashboard → Edge Functions → Select function → Logs
- Check for errors, timeouts, or unexpected behavior

### Database Monitoring
- Supabase Dashboard → Database → Monitor
- Track query performance
- Monitor storage usage
- Review slow queries

### Lovable AI Usage
Monitor AI request usage:
- Lovable Dashboard → Settings → Usage
- Track costs and rate limits
- Optimize prompts if needed

---

## 10. Custom Domain Setup

### Using Lovable Custom Domains
1. Navigate to Project → Settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed
4. Update Supabase authentication URLs to use custom domain

### DNS Configuration
Point your domain to Lovable:
- CNAME record: `your-domain.com` → `your-project.lovable.app`
- Wait for DNS propagation (can take up to 48 hours)

### Update Authentication URLs
After custom domain is active:
1. Update Site URL in Supabase to custom domain
2. Add custom domain to Redirect URLs
3. Test authentication flow thoroughly

---

## 11. Post-Deployment Verification

### Functional Testing
- [ ] User registration works
- [ ] Login/logout works
- [ ] Onboarding flow completes
- [ ] Mood tracking saves to database
- [ ] AI analysis provides results
- [ ] Goals generation works
- [ ] Recommendations display
- [ ] Crisis support accessible
- [ ] Data export works
- [ ] PWA installs on mobile

### Performance Testing
- [ ] Page load times < 3 seconds
- [ ] AI responses < 5 seconds
- [ ] Database queries < 500ms
- [ ] Service worker caching works
- [ ] Offline mode functions

### Security Testing
- [ ] Unauthorized access blocked
- [ ] User data properly isolated
- [ ] Crisis events logged
- [ ] Password requirements enforced
- [ ] Session timeout works

---

## 12. Troubleshooting

### Common Deployment Issues

**Problem**: "Requested path is invalid" error on login
**Solution**: Update Site URL and Redirect URLs in Supabase Authentication → URL Configuration

**Problem**: Edge functions not working
**Solution**: Check Supabase Dashboard → Edge Functions → Logs for errors. Verify secrets are configured.

**Problem**: PWA not installing
**Solution**: Verify HTTPS is enabled, check manifest.json is accessible, test on supported browsers

**Problem**: AI analysis failing
**Solution**: Check Lovable AI usage limits, verify LOVABLE_API_KEY is set, check edge function logs

**Problem**: Database connection errors
**Solution**: Verify Supabase project is active, check RLS policies, verify user is authenticated

---

## 13. Rollback Procedure

If deployment issues occur:

1. **Immediate Rollback**:
   - Lovable maintains version history
   - Click project name → Settings → History
   - Revert to previous working version

2. **Database Rollback**:
   - Supabase maintains daily backups
   - Contact Supabase support for restoration if needed
   - Database changes are tracked in migrations

3. **Edge Function Rollback**:
   - Edge functions auto-deploy with code changes
   - Revert code to restore previous function versions

---

## 14. Support & Resources

### Documentation

- [Supabase Documentation](https://supabase.com/docs)
- [PWA Documentation](https://web.dev/progressive-web-apps/)

### Support Channels
- Supabase Support: https://supabase.com/support
- Community: Lovable Discord, Supabase Discord

### Emergency Contacts
For production emergencies:
1. Check Supabase status page
2. Review Lovable status page
3. Contact support channels above

---

## 15. Cost Considerations

### Pricing
- Check current plan limits in Lovable dashboard
- Monitor project usage
- Plan for scaling needs

### Supabase Costs
- Free tier: Limited resources
- Pro tier: Recommended for production
- Monitor usage: Database, Edge Functions, Storage, Bandwidth

###  AI Costs
- Usage-based pricing
- Free tier included
- Monitor request volume
- Optimize AI prompts for efficiency

---

## 16. Legal & Compliance

### HIPAA Requirements (if handling PHI)
- [ ] Business Associate Agreement (BAA) with Supabase
- [ ] Privacy Policy published
- [ ] Terms of Service published
- [ ] User consent obtained
- [ ] Data retention policy defined
- [ ] Incident response plan created

### General Compliance
- [ ] GDPR compliance (if serving EU users)
- [ ] CCPA compliance (if serving California users)
- [ ] Accessibility compliance (WCAG 2.1 AA)
- [ ] Cookie policy (if using cookies)

---

## Document Version
- **Version**: 1.0
- **Date**: 2025-10-01
- **Last Updated**: 2025-10-01
