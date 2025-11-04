import React, { useMemo } from 'react'
import styled from 'styled-components'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
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

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: ${theme.spacing.lg};
  margin: ${theme.spacing.xl} 0;
`

const MetricCard = styled.div`
  background: linear-gradient(135deg, ${theme.colors.purple}, ${theme.colors.teal});
  padding: ${theme.spacing.lg};
  border-radius: ${theme.borderRadius.md};
  text-align: center;
  border: 2px solid ${theme.colors.mint};
  box-shadow: ${theme.shadows.md};
  transition: transform ${theme.transitions.normal};
  
  &:hover {
    transform: translateY(-5px);
  }
`

const MetricValue = styled.div`
  font-size: 2.5rem;
  color: ${theme.colors.light};
  font-weight: bold;
  margin: ${theme.spacing.md} 0;
`

const MetricLabel = styled.div`
  color: ${theme.colors.light};
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 1px;
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

const KeyFinding = styled.div`
  background: linear-gradient(135deg, rgba(87, 82, 196, 0.2), rgba(71, 105, 117, 0.2));
  padding: ${theme.spacing.lg};
  border-radius: ${theme.borderRadius.md};
  margin: ${theme.spacing.lg} 0;
  border-left: 5px solid ${theme.colors.mint};
`

const FindingTitle = styled.h4`
  color: ${theme.colors.mint};
  font-size: 1.2rem;
  margin-bottom: ${theme.spacing.sm};
`

const FindingText = styled.p`
  color: ${theme.colors.light};
  line-height: 1.8;
`

const Overview = ({ data }) => {
  const metrics = useMemo(() => {
    if (!data) return null
    
    const totalAwards = data.length
    const uniqueLaureates = new Set(data.map(d => d.laureate_id)).size
    const femaleCount = data.filter(d => d.sex === 'female').length
    const maleCount = data.filter(d => d.sex === 'male').length
    const femalePercentage = ((femaleCount / (femaleCount + maleCount)) * 100).toFixed(2)
    const sharedPrizes = data.filter(d => d.is_shared === 1).length
    const sharedPercentage = ((sharedPrizes / totalAwards) * 100).toFixed(1)
    const countries = new Set(data.map(d => d.birth_country).filter(Boolean)).size
    const avgAge = (data.filter(d => d.age_at_award).reduce((sum, d) => sum + (d.age_at_award || 0), 0) / data.filter(d => d.age_at_award).length).toFixed(1)
    
    return {
      totalAwards,
      uniqueLaureates,
      femalePercentage,
      sharedPercentage,
      countries,
      avgAge,
      femaleCount,
      maleCount
    }
  }, [data])

  const categoryData = useMemo(() => {
    if (!data) return []
    const categories = {}
    data.forEach(d => {
      categories[d.category] = (categories[d.category] || 0) + 1
    })
    return Object.entries(categories).map(([name, value]) => ({ name, value }))
  }, [data])

  const genderData = useMemo(() => {
    if (!data || !metrics) return []
    return [
      { name: 'Male', value: metrics.maleCount },
      { name: 'Female', value: metrics.femaleCount }
    ]
  }, [data, metrics])

  if (!data || !metrics) return null

  return (
    <Section>
      <SectionTitle>Executive Overview</SectionTitle>
      
      <MetricsGrid>
        <MetricCard>
          <MetricLabel>Total Awards</MetricLabel>
          <MetricValue>{metrics.totalAwards}</MetricValue>
        </MetricCard>
        <MetricCard>
          <MetricLabel>Unique Laureates</MetricLabel>
          <MetricValue>{metrics.uniqueLaureates}</MetricValue>
        </MetricCard>
        <MetricCard>
          <MetricLabel>Female Representation</MetricLabel>
          <MetricValue>{metrics.femalePercentage}%</MetricValue>
        </MetricCard>
        <MetricCard>
          <MetricLabel>Shared Prizes</MetricLabel>
          <MetricValue>{metrics.sharedPercentage}%</MetricValue>
        </MetricCard>
        <MetricCard>
          <MetricLabel>Countries</MetricLabel>
          <MetricValue>{metrics.countries}</MetricValue>
        </MetricCard>
        <MetricCard>
          <MetricLabel>Avg Age at Award</MetricLabel>
          <MetricValue>{metrics.avgAge}</MetricValue>
        </MetricCard>
      </MetricsGrid>

      <ChartContainer>
        <ChartTitle>Awards by Category</ChartTitle>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={categoryData}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.teal} opacity={0.3} />
            <XAxis dataKey="name" stroke={theme.colors.light} angle={-45} textAnchor="end" height={100} />
            <YAxis stroke={theme.colors.light} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: theme.colors.dark, 
                border: `2px solid ${theme.colors.teal}`,
                borderRadius: theme.borderRadius.sm,
                color: theme.colors.light
              }} 
            />
            <Bar dataKey="value" fill={theme.colors.purple} />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>

      <ChartContainer>
        <ChartTitle>Gender Distribution</ChartTitle>
        <ResponsiveContainer width="100%" height={400}>
          <PieChart>
            <Pie
              data={genderData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
              outerRadius={150}
              fill="#8884d8"
              dataKey="value"
            >
              {genderData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={index === 0 ? theme.colors.teal : theme.colors.purple} />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ 
                backgroundColor: theme.colors.dark, 
                border: `2px solid ${theme.colors.teal}`,
                borderRadius: theme.borderRadius.sm,
                color: theme.colors.light
              }} 
            />
          </PieChart>
        </ResponsiveContainer>
      </ChartContainer>

      <KeyFinding>
        <FindingTitle>Critical Finding: Gender Disparity</FindingTitle>
        <FindingText>
          Despite 125 years of Nobel Prize history, female representation remains at {metrics.femalePercentage}%. 
          This systematic underrepresentation persists even in the modern era, indicating structural barriers 
          that require immediate institutional intervention. Physics (2.10%) and Economic Sciences (1.89%) 
          show the most severe underrepresentation.
        </FindingText>
      </KeyFinding>

      <KeyFinding>
        <FindingTitle>Collaboration Revolution</FindingTitle>
        <FindingText>
          {metrics.sharedPercentage}% of Nobel Prizes are now shared, indicating a fundamental transformation 
          in scientific practice from individual endeavor to collaborative enterprise. Modern research 
          increasingly requires diverse expertise and large teams, particularly in Physics and Medicine.
        </FindingText>
      </KeyFinding>

      <KeyFinding>
        <FindingTitle>Global Distribution</FindingTitle>
        <FindingText>
          Nobel laureates come from {metrics.countries} countries, but research excellence remains highly 
          concentrated. The top 3 countries (USA, UK, Germany) account for 46.1% of all awards, reflecting 
          historical patterns of research investment and institutional prestige.
        </FindingText>
      </KeyFinding>
    </Section>
  )
}

export default Overview
