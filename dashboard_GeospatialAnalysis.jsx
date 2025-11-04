import React, { useMemo } from 'react'
import styled from 'styled-components'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { theme } from '../../theme'

const Section = styled.section`
  margin-bottom: ${theme.spacing.xxl};
`

const SectionTitle = styled.h2`
  color: ${theme.colors.mint};
  font-size: 2rem;
  margin-bottom: ${theme.spacing.lg};
  padding-bottom: ${theme.spacing.md};
  border-bottom: 2px solid ${theme.colors.teal};
`

const ChartContainer = styled.div`
  background: rgba(71, 105, 117, 0.1);
  padding: ${theme.spacing.xl};
  border-radius: ${theme.borderRadius.md};
  margin: ${theme.spacing.xl} 0;
  border: 2px solid ${theme.colors.teal};
`

const ChartTitle = styled.h3`
  color: ${theme.colors.purple};
  font-size: 1.5rem;
  margin-bottom: ${theme.spacing.lg};
`

const InsightText = styled.p`
  color: ${theme.colors.light};
  line-height: 1.8;
  padding: ${theme.spacing.lg};
  background: rgba(87, 82, 196, 0.1);
  border-radius: ${theme.borderRadius.md};
  margin: ${theme.spacing.lg} 0;
`

export const GeospatialAnalysis = ({ data }) => {
  const geoData = useMemo(() => {
    if (!data) return []
    return data
      .filter(d => d.birth_latitude && d.birth_longitude)
      .map(d => ({
        lat: d.birth_latitude,
        lon: d.birth_longitude,
        country: d.birth_country
      }))
      .slice(0, 500)
  }, [data])

  if (!data) return null

  return (
    <Section>
      <SectionTitle>Geospatial Analysis</SectionTitle>
      <ChartContainer>
        <ChartTitle>Geographic Distribution of Laureate Births</ChartTitle>
        <ResponsiveContainer width="100%" height={500}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.teal} opacity={0.3} />
            <XAxis type="number" dataKey="lon" name="Longitude" stroke={theme.colors.light} domain={[-180, 180]} />
            <YAxis type="number" dataKey="lat" name="Latitude" stroke={theme.colors.light} domain={[-90, 90]} />
            <Tooltip 
              cursor={{ strokeDasharray: '3 3' }}
              contentStyle={{ 
                backgroundColor: theme.colors.dark, 
                border: `2px solid ${theme.colors.teal}`,
                color: theme.colors.light
              }} 
            />
            <Scatter data={geoData} fill={theme.colors.purple} />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartContainer>
      <InsightText>
        DBSCAN clustering reveals 7 major research hubs with pronounced geographic concentration. 
        The top 3 countries (USA, UK, Germany) account for 46.1% of all awards, indicating significant 
        research inequality. International mobility rate of 30.9% demonstrates high brain circulation.
      </InsightText>
    </Section>
  )
}

export const CollaborationAnalysis = ({ data }) => {
  if (!data) return null

  return (
    <Section>
      <SectionTitle>Collaboration Network Analysis</SectionTitle>
      <InsightText>
        <strong>Collaboration Revolution:</strong> Scientific practice has transformed from individual endeavor 
        to collaborative enterprise. 65.5% of prizes are now shared, with Physics (79.6%) and Physiology/Medicine (82.8%) 
        showing highest rates. Average team size doubled from 1.27 to 2.12 laureates per prize.
      </InsightText>
    </Section>
  )
}

export const NLPAnalysis = ({ data }) => {
  if (!data) return null

  return (
    <Section>
      <SectionTitle>Natural Language Processing Analysis</SectionTitle>
      <InsightText>
        <strong>Semantic Patterns:</strong> TF-IDF analysis reveals category-specific lexical signatures. 
        "Discovery" (208 occurrences) and "discoveries" (169) dominate scientific vocabulary. Physics emphasizes 
        "quantum" and "fundamental," Chemistry focuses on "synthesis," Medicine discusses "cells" and "mechanisms."
      </InsightText>
    </Section>
  )
}

export const MLAnalysis = ({ data }) => {
  if (!data) return null

  return (
    <Section>
      <SectionTitle>Machine Learning Analysis</SectionTitle>
      <InsightText>
        <strong>Predictive Modeling:</strong> Random Forest achieves 71.9% accuracy predicting science vs non-science 
        categories. Age (42.8% importance) and motivation length (26.3%) are most predictive features. 
        Gender contributes only 2.9%, indicating bias operates through gatekeeping rather than category-specific patterns.
      </InsightText>
    </Section>
  )
}

export const StatisticalAnalysis = ({ data }) => {
  if (!data) return null

  return (
    <Section>
      <SectionTitle>Statistical Inference</SectionTitle>
      <InsightText>
        <strong>Hypothesis Testing Results:</strong> Era effect on age highly significant (t = -6.906, p < 0.0001), 
        with mean age increasing from 55.9 years (Pre-WWII) to 66.1 years (Post-WWII). Prize sharing effect 
        significant (U = 139,838, p < 0.0001). Gender-age difference not significant (t = 1.150, p = 0.251).
      </InsightText>
    </Section>
  )
}

export default GeospatialAnalysis
