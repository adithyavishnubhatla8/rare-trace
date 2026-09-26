-- =============================================================================
-- RARETRACE: Rare-Disease Case Identification Using DBSCAN Anomaly Detection
-- Supabase PostgreSQL Database Schema
-- =============================================================================
-- Run this script in your Supabase SQL Editor:
-- https://supabase.com/dashboard/project/<your-project-id>/sql/new

-- 1. Create the analyses table
CREATE TABLE IF NOT EXISTS public.analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id TEXT UNIQUE NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_filename TEXT NOT NULL,
    source TEXT DEFAULT 'User Upload',
    description TEXT,
    total_records INTEGER NOT NULL,
    total_clusters INTEGER NOT NULL,
    anomalies INTEGER NOT NULL,
    anomaly_percentage DOUBLE PRECISION NOT NULL,
    eps DOUBLE PRECISION,
    min_samples INTEGER,
    silhouette_score DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'completed',
    results_summary JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Performance indexes
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON public.analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_id ON public.analyses(analysis_id);

-- 3. Enable Row-Level Security (RLS)
ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;

-- 4. Create RLS access policies for the API
-- Allow public reading of analysis history
CREATE POLICY "Allow public read access"
ON public.analyses
FOR SELECT
USING (true);

-- Allow server-side API insertions using anon/publishable key
CREATE POLICY "Allow public insert access"
ON public.analyses
FOR INSERT
WITH CHECK (true);

-- Allow updates to analysis status
CREATE POLICY "Allow public update access"
ON public.analyses
FOR UPDATE
USING (true)
WITH CHECK (true);

-- 5. Optional verification query
COMMENT ON TABLE public.analyses IS 'Stores RARETRACE machine learning analysis metadata, cluster metrics, and DBSCAN parameters without storing sensitive patient PII.';
