
# Deployment Guide

## Prerequisites
- Node.js 18+
- Supabase account
- Vercel account (optional)

## Environment Setup
1. Copy `.env.production` and update with your values
2. Configure Supabase project
3. Run database migrations

## Build and Deploy
```bash
npm install
npm run build
npm run preview  # Test production build
```

## Production Checklist
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup strategy implemented
