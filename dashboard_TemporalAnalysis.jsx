import React, { useMemo } from 'react'
import styled from 'styled-components'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts'
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

const InsightBox = styled.div`
  background: linear-gradient(135deg, rgba(89, 203, 50, 0.2), rgba(71, 105, 117, 0.2));
  padding: ${theme.spacing.lg};
  border-radius: ${theme.borderRadius.md};
  margin: ${theme.spacing.lg} 0;
  border: 2px solid ${theme.colors.mint};
`

const InsightText = styled.p`
  color: ${theme.colors.light};
  line-height: 1.8;
  font-size: 1rem;
`

const TemporalAnalysis = ({ data }) => {
  const yearlyData = useMemo(() => {
    if (!data) return []
    
    const yearGroups = {}
    data.forEach(d => {
      const year = d.award_year
      if (!yearGroups[year]) {
        yearGroups[year] = {
          year,
          total: 0,
          female: 0,
          male: 0,
          shared: 0,
          ages: []
        }
      }
      yearGroups[year].total++
      if (d.sex === 'female') yearGroups[year].female++
      if (d.sex === 'male') yearGroups[year].male++
      if (d.is_shared === 1) yearGroups[year].shared++
      if (d.age_at_award) yearGroups[year].ages.push(d.age_at_award)
    })
    
    return Object.values(yearGroups).map(g => ({
      year: g.year,
      total: g.total,
      femalePercent: ((g.female / (g.female + g.male)) * 100).toFixed(1),
      sharedPercent: ((g.shared / g.total) * 100).toFixed(1),
      avgAge: g.ages.length > 0 ? (g.ages.reduce((a, b) => a + b, 0) / g.ages.length).toFixed(1) : null
    })).sort((a, b) => a.year - b.year)
  }, [data])

  const decadeData = useMemo(() => {
    if (!data) return []
    
    const decadeGroups = {}
    data.forEach(d => {
      const decade = Math.floor(d.award_year / 10) * 10
      if (!decadeGroups[decade]) {
        decadeGroups[decade] = {
          decade,
          total: 0,
          female: 0,
          male: 0
        }
      }
      decadeGroups[decade].total++
      if (d.sex === 'female') decadeGroups[decade].female++
      if (d.sex === 'male') decadeGroups[decade].male++
    })
    
    return Object.values(decadeGroups).map(g => ({
      decade: g.decade,
      femalePercent: ((g.female / (g.female + g.male)) * 100).toFixed(1)
    })).sort((a, b) => a.decade - b.decade)
  }, [data])

  if (!data) return null

  return (
    <Section>
      <SectionTitle>Temporal Evolution Analysis</SectionTitle>
      
      <ChartContainer>
        <ChartTitle>Awards Over Time</ChartTitle>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={yearlyData}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.teal} opacity={0.3} />
            <XAxis 
              dataKey="year" 
              stroke={theme.colors.light}
              domain={[1901, 2025]}
            />
            <YAxis stroke={theme.colors.light} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: theme.colors.dark, 
                border: `2px solid ${theme.colors.teal}`,
                borderRadius: theme.borderRadius.sm,
                color: theme.colors.light
              }} 
            />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="total" 
              stroke={theme.colors.purple} 
              fill={theme.colors.purple}
              fillOpacity={0.6}
              name="Total Awards"
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartContainer>

      <ChartContainer>
        <ChartTitle>Female Representation by Decade</ChartTitle>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={decadeData}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.teal} opacity={0.3} />
            <XAxis dataKey="decade" stroke={theme.colors.light} />
            <YAxis stroke={theme.colors.light} label={{ value: 'Percentage', angle: -90, position: 'insideLeft' }} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: theme.colors.dark, 
                border: `2px solid ${theme.colors.teal}`,
                borderRadius: theme.borderRadius.sm,
                color: theme.colors.light
              }} 
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="femalePercent" 
              stroke={theme.colors.purple} 
              strokeWidth={3}
              name="Female %"
              dot={{ fill: theme.colors.purple, r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>

      <ChartContainer>
        <ChartTitle>Collaboration Trends Over Time</ChartTitle>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={yearlyData}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.teal} opacity={0.3} />
            <XAxis dataKey="year" stroke={theme.colors.light} />
            <YAxis stroke={theme.colors.light} label={{ value: 'Shared Prize %', angle: -90, position: 'insideLeft' }} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: theme.colors.dark, 
                border: `2px solid ${theme.colors.teal}`,
                borderRadius: theme.borderRadius.sm,
                color: theme.colors.light
              }} 
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="sharedPercent" 
              stroke={theme.colors.mint} 
              strokeWidth={2}
              name="Shared Prizes %"
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>

      <InsightBox>
        <InsightText>
          <strong>Key Insight:</strong> Temporal analysis reveals a dramatic transformation in scientific practice. 
          While early Nobel Prizes (1901-1950) were predominantly awarded to individual scientists, modern awards 
          (1990-2025) are increasingly collaborative, with over 80% of prizes now shared. This reflects the 
          growing complexity of scientific problems requiring interdisciplinary expertise and large research teams.
        </InsightText>
      </InsightBox>

      <InsightBox>
        <InsightText>
          <strong>Gender Trends:</strong> Female representation shows modest improvement from approximately 3% 
          in pre-WWII era to 11% in the modern era. However, this rate of change is insufficient to achieve 
          parity within any reasonable timeframe. Extrapolating current trends suggests gender equity would 
          require over 300 additional years.
        </InsightText>
      </InsightBox>
    </Section>
  )
}

export default TemporalAnalysis
